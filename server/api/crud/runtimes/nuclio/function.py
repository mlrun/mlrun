# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import base64
import shlex

import nuclio
import nuclio.utils
import requests

import mlrun
import mlrun.common.constants
import mlrun.common.schemas
import mlrun.datastore
import mlrun.errors
import mlrun.runtimes.nuclio.function
import mlrun.runtimes.pod
import mlrun.utils
import server.api.crud.runtimes.nuclio.helpers
import server.api.runtime_handlers
import server.api.utils.builder
import server.api.utils.singletons.k8s
from mlrun.utils import logger


def deploy_nuclio_function(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
    auth_info: mlrun.common.schemas.AuthInfo = None,
    client_version: str = None,
    builder_env: dict = None,
    client_python_version: str = None,
):
    """Deploys a nuclio function.

    :param function:              nuclio function object
    :param auth_info:             service AuthInfo
    :param client_version:        mlrun client version
    :param builder_env:           mlrun builder environment (for config/credentials)
    :param client_python_version: mlrun client python version
    """
    function_name, project_name, function_config = _compile_function_config(
        function,
        client_version=client_version,
        client_python_version=client_python_version,
        builder_env=builder_env or {},
        auth_info=auth_info,
    )

    # if mode allows it, enrich function http trigger with an ingress
    server.api.crud.runtimes.nuclio.helpers.enrich_function_with_ingress(
        function_config,
        function.spec.add_templated_ingress_host_mode
        or mlrun.mlconf.httpdb.nuclio.add_templated_ingress_host_mode,
        function.spec.service_type or mlrun.mlconf.httpdb.nuclio.default_service_type,
    )

    try:
        logger.info(
            "Starting Nuclio function deployment",
            function_name=function_name,
            project_name=project_name,
        )
        return nuclio.deploy.deploy_config(
            function_config,
            dashboard_url=mlrun.mlconf.nuclio_dashboard_url,
            name=function_name,
            project=project_name,
            tag=function.metadata.tag,
            verbose=function.verbose,
            create_new=mlrun.mlconf.httpdb.projects.leader == "mlrun",
            watch=False,
            return_address_mode=nuclio.deploy.ReturnAddressModes.all,
            auth_info=auth_info.to_nuclio_auth_info() if auth_info else None,
        )
    except nuclio.utils.DeployError as exc:
        if exc.err:
            err_message = (
                f"Failed to deploy nuclio function {project_name}/{function_name}"
            )

            try:
                # the error might not be jsonable, so we'll try to parse it
                # and extract the error message
                json_err = exc.err.response.json()
                if "error" in json_err:
                    err_message += f" {json_err['error']}"
                if "errorStackTrace" in json_err:
                    logger.warning(
                        "Failed to deploy nuclio function",
                        nuclio_stacktrace=json_err["errorStackTrace"],
                    )
            except Exception as parse_exc:
                logger.warning(
                    "Failed to parse nuclio deploy error",
                    parse_exc=mlrun.errors.err_to_str(parse_exc),
                )

            mlrun.errors.raise_for_status(
                exc.err.response,
                err_message,
            )
        raise


def get_nuclio_deploy_status(
    name,
    project,
    tag,
    last_log_timestamp=0,
    verbose=False,
    resolve_address=True,
    auth_info: mlrun.common.schemas.AuthInfo = None,
):
    """
    Get nuclio function deploy status

    :param name:                function name
    :param project:             project name
    :param tag:                 function tag
    :param last_log_timestamp:  last log timestamp
    :param verbose:             print logs
    :param resolve_address:     whether to resolve function address
    :param auth_info:           authentication information
    """
    api_address = nuclio.deploy.find_dashboard_url(mlrun.mlconf.nuclio_dashboard_url)
    name = mlrun.runtimes.nuclio.function.get_fullname(name, project, tag)
    get_err_message = f"Failed to get function {name} deploy status"

    try:
        (
            state,
            address,
            last_log_timestamp,
            outputs,
            function_status,
        ) = nuclio.deploy.get_deploy_status(
            api_address,
            name,
            last_log_timestamp,
            verbose,
            resolve_address,
            return_function_status=True,
            auth_info=auth_info.to_nuclio_auth_info() if auth_info else None,
        )
    except requests.exceptions.ConnectionError as exc:
        mlrun.errors.raise_for_status(
            exc.response,
            get_err_message,
        )

    except nuclio.utils.DeployError as exc:
        if exc.err:
            mlrun.errors.raise_for_status(
                exc.err.response,
                get_err_message,
            )
        raise exc
    else:
        text = "\n".join(outputs) if outputs else ""
        return state, address, name, last_log_timestamp, text, function_status


def pure_nuclio_deployed_restricted():
    """
    Decorator to restrict the usage of the decorated function to pure nuclio deployed runtimes only.
    Pure nuclio deployed runtimes are runtimes that their images are not built by MLRun, but are built and deployed
    completely by nuclio.
    """

    def decorator(callback):
        def wrapper(function, *args, **kwargs):
            if (
                function.kind
                not in mlrun.runtimes.RuntimeKinds.pure_nuclio_deployed_runtimes()
            ):
                return

            return callback(function, *args, **kwargs)

        return wrapper

    return decorator


def _compile_function_config(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
    client_version: str = None,
    client_python_version: str = None,
    builder_env=None,
    auth_info=None,
):
    _set_function_labels(function)

    # resolve env vars before compiling the nuclio spec, as we need to set them in the spec
    env_dict, external_source_env_dict = _resolve_env_vars(function)

    project = function.metadata.project or mlrun.mlconf.default_project
    tag = function.metadata.tag

    serving_spec_volume = None
    serving_spec = function._get_serving_spec()
    if serving_spec is not None:
        # since environment variables have a limited size,
        # large serving specs are stored in config maps that are mounted to the pod
        if len(serving_spec) >= mlrun.mlconf.httpdb.nuclio.serving_spec_env_cutoff:
            function_name = mlrun.runtimes.nuclio.function.get_fullname(
                function.metadata.name, project, tag
            )
            k8s_helper = server.api.utils.singletons.k8s.get_k8s_helper()
            confmap_name = k8s_helper.ensure_configmap(
                mlrun.common.constants.MLRUN_MODEL_CONF,
                function_name,
                {mlrun.common.constants.MLRUN_SERVING_SPEC_FILENAME: serving_spec},
                labels={mlrun.common.constants.MLRUN_CREATED_LABEL: "true"},
            )
            volume_name = mlrun.common.constants.MLRUN_MODEL_CONF
            volume_mount = {
                "name": volume_name,
                "mountPath": mlrun.common.constants.MLRUN_SERVING_SPEC_MOUNT_PATH,
                "readOnly": True,
            }

            serving_spec_volume = {
                "volume": {"name": volume_name, "configMap": {"name": confmap_name}},
                "volumeMount": volume_mount,
            }
        else:
            env_dict["SERVING_SPEC_ENV"] = serving_spec

    # resolve sidecars images
    sidecars = function.spec.config.get("spec.sidecars") or []
    for sidecar in sidecars:
        sidecar_image = sidecar.get("image")
        if sidecar_image:
            sidecar["image"] = server.api.utils.builder.resolve_and_enrich_image_target(
                sidecar_image,
                client_version=client_version,
                client_python_version=client_python_version,
            )

    nuclio_spec = nuclio.ConfigSpec(
        env=env_dict,
        external_source_env=external_source_env_dict,
        config=function.spec.config,
    )

    _resolve_and_set_build_requirements_and_commands(function, nuclio_spec)
    _resolve_and_set_nuclio_runtime(
        function, nuclio_spec, client_version, client_python_version
    )

    handler = function.spec.function_handler

    _set_build_params(function, nuclio_spec, builder_env, project, auth_info)
    _set_function_scheduling_params(function, nuclio_spec)
    _set_function_replicas(function, nuclio_spec)
    _set_misc_specs(function, nuclio_spec)

    # if the user code is given explicitly or from a source, we need to set the handler and relevant attributes
    if (
        function.spec.base_spec
        or function.spec.build.functionSourceCode
        or function.spec.build.source
        or function.kind == mlrun.runtimes.RuntimeKinds.serving  # serving can be empty
    ):
        config = function.spec.base_spec
        if not config:
            # if base_spec was not set (when not using code_to_function) and we have base64 code
            # we create the base spec with essential attributes
            config = nuclio.config.new_config()
            mlrun.utils.update_in(config, "spec.handler", handler or "main:handler")

        config = nuclio.config.extend_config(
            config, nuclio_spec, tag, function.spec.build.code_origin
        )

        if (
            function.kind == mlrun.runtimes.RuntimeKinds.serving
            and not mlrun.utils.get_in(config, "spec.build.functionSourceCode")
        ):
            _set_source_code_and_handler(function, config)
    else:
        # this may also be called in case of using single file code_to_function(embed_code=False)
        # this option need to be removed or be limited to using remote files (this code runs in server)
        function_name, config, code = nuclio.build_file(
            function.spec.source,
            name=function.metadata.name,
            project=project,
            handler=handler,
            tag=tag,
            spec=nuclio_spec,
            kind=function.spec.function_kind,
            verbose=function.verbose,
        )

    mlrun.utils.update_in(
        config, "spec.volumes", function.spec.generate_nuclio_volumes()
    )

    _resolve_and_set_base_image(function, config, client_version, client_python_version)
    function_name = _set_function_name(function, config, project, tag)

    if serving_spec_volume is not None:
        mlrun.utils.update_in(config, "spec.volumes", serving_spec_volume, append=True)

    return function_name, project, config


def _set_function_labels(function):
    labels = function.metadata.labels or {}
    labels.update({"mlrun/class": function.kind})
    for key, value in labels.items():
        # Adding escaping to the key to prevent it from being split by dots if it contains any
        function.set_config(f"metadata.labels.\\{key}\\", value)


def _resolve_env_vars(function):
    # Add secret configurations to function's pod spec, if secret sources were added.
    # Needs to be here, since it adds env params, which are handled in the next lines.
    # This only needs to run if we're running within k8s context. If running in Docker, for example, skip.
    if server.api.utils.singletons.k8s.get_k8s_helper(
        silent=True
    ).is_running_inside_kubernetes_cluster():
        _add_secrets_config_to_function_spec(function)

    env_dict, external_source_env_dict = function._get_nuclio_config_spec_env()

    # In nuclio 1.6.0<=v<1.8.0, python runtimes default behavior was to not decode event strings
    # Our code is counting on the strings to be decoded, so add the needed env var for those versions
    if (
        server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
            "1.6.0", "1.8.0"
        )
        and "NUCLIO_PYTHON_DECODE_EVENT_STRINGS" not in env_dict
    ):
        env_dict["NUCLIO_PYTHON_DECODE_EVENT_STRINGS"] = "true"

    return env_dict, external_source_env_dict


def _resolve_and_set_nuclio_runtime(
    function, nuclio_spec, client_version, client_python_version
):
    nuclio_runtime = (
        function.spec.nuclio_runtime
        or server.api.crud.runtimes.nuclio.helpers.resolve_nuclio_runtime_python_image(
            mlrun_client_version=client_version, python_version=client_python_version
        )
    )

    # For backwards compatibility, we need to adjust the runtime for old Nuclio versions
    if server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
        "0.0.0", "1.6.0"
    ) and nuclio_runtime in [
        "python:3.7",
        "python:3.8",
        "python:3.9",
    ]:
        nuclio_runtime_set_from_spec = nuclio_runtime == function.spec.nuclio_runtime
        if nuclio_runtime_set_from_spec:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Nuclio version does not support the configured runtime: {nuclio_runtime}"
            )
        else:
            # our default is python:3.9, simply set it to python:3.6 to keep supporting envs with old Nuclio
            nuclio_runtime = "python:3.6"

    nuclio_spec.set_config("spec.runtime", nuclio_runtime)


@pure_nuclio_deployed_restricted()
def _resolve_and_set_build_requirements_and_commands(function, nuclio_spec):
    nuclio_spec.cmd = function.spec.build.commands or []
    _resolve_and_set_build_requirements(function, nuclio_spec)


def _resolve_and_set_build_requirements(function, nuclio_spec):
    if function.spec.build.requirements:
        resolved_requirements = []
        # wrap in single quote to ensure that the requirement is treated as a single string
        # quote the requirement to avoid issues with special characters, double quotes, etc.
        for requirement in function.spec.build.requirements:
            # -r / --requirement are flags and should not be escaped
            # we allow such flags (could be passed within the requirements.txt file) and do not
            # try to open the file and include its content since it might be a remote file
            # given on the base image.
            for req_flag in ["-r", "--requirement"]:
                if requirement.startswith(req_flag):
                    requirement = requirement[len(req_flag) :].strip()
                    resolved_requirements.append(req_flag)
                    break

            resolved_requirements.append(shlex.quote(requirement))

        encoded_requirements = " ".join(resolved_requirements)
        nuclio_spec.cmd.append(f"python -m pip install {encoded_requirements}")


def _set_build_params(function, nuclio_spec, builder_env, project, auth_info=None):
    # handle archive build params
    if function.spec.build.source:
        server.api.crud.runtimes.nuclio.helpers.compile_nuclio_archive_config(
            nuclio_spec, function, builder_env, project, auth_info=auth_info
        )

    if function.spec.no_cache:
        nuclio_spec.set_config("spec.build.noCache", True)
    if function.spec.build.functionSourceCode:
        nuclio_spec.set_config(
            "spec.build.functionSourceCode", function.spec.build.functionSourceCode
        )

    image_pull_secret = (
        server.api.crud.runtimes.nuclio.helpers.resolve_function_image_pull_secret(
            function
        )
    )
    if image_pull_secret:
        nuclio_spec.set_config("spec.imagePullSecrets", image_pull_secret)

    if function.spec.base_image_pull:
        nuclio_spec.set_config("spec.build.noBaseImagesPull", False)

    if function.spec.build.extra_args:
        nuclio_spec.set_config(
            "spec.build.flags",
            server.api.crud.runtimes.nuclio.helpers.parse_extra_args_to_nuclio_build_flags(
                function.spec.build.extra_args
            ),
        )


def _set_function_scheduling_params(function, nuclio_spec):
    # don't send node selections if nuclio is not compatible
    if mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
        "1.5.20", "1.6.10"
    ):
        if function.spec.node_selector:
            nuclio_spec.set_config("spec.nodeSelector", function.spec.node_selector)
        if function.spec.node_name:
            nuclio_spec.set_config("spec.nodeName", function.spec.node_name)
        if function.spec.affinity:
            nuclio_spec.set_config(
                "spec.affinity",
                mlrun.runtimes.pod.get_sanitized_attribute(function.spec, "affinity"),
            )

    # don't send tolerations if nuclio is not compatible
    if mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility("1.7.5"):
        if function.spec.tolerations:
            nuclio_spec.set_config(
                "spec.tolerations",
                mlrun.runtimes.pod.get_sanitized_attribute(
                    function.spec, "tolerations"
                ),
            )
    # don't send preemption_mode if nuclio is not compatible
    if mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility("1.8.6"):
        if function.spec.preemption_mode:
            nuclio_spec.set_config(
                "spec.PreemptionMode",
                function.spec.preemption_mode,
            )


def _set_function_replicas(function, nuclio_spec):
    if function.spec.replicas:
        nuclio_spec.set_config(
            "spec.minReplicas",
            mlrun.utils.as_number("spec.Replicas", function.spec.replicas),
        )
        nuclio_spec.set_config(
            "spec.maxReplicas",
            mlrun.utils.as_number("spec.Replicas", function.spec.replicas),
        )
    else:
        nuclio_spec.set_config(
            "spec.minReplicas",
            mlrun.utils.as_number("spec.minReplicas", function.spec.min_replicas),
        )
        nuclio_spec.set_config(
            "spec.maxReplicas",
            mlrun.utils.as_number("spec.maxReplicas", function.spec.max_replicas),
        )


def _set_misc_specs(function, nuclio_spec):
    # in Nuclio >= 1.6.x default serviceType has changed to "ClusterIP".
    nuclio_spec.set_config(
        "spec.serviceType",
        function.spec.service_type or mlrun.mlconf.httpdb.nuclio.default_service_type,
    )
    if function.spec.readiness_timeout:
        nuclio_spec.set_config(
            "spec.readinessTimeoutSeconds", function.spec.readiness_timeout
        )
    if function.spec.readiness_timeout_before_failure:
        nuclio_spec.set_config(
            "spec.waitReadinessTimeoutBeforeFailure",
            function.spec.readiness_timeout_before_failure,
        )
    if function.spec.resources:
        nuclio_spec.set_config("spec.resources", function.spec.resources)

    # don't send default or any priority class name if nuclio is not compatible
    if (
        function.spec.priority_class_name
        and mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.6.18"
        )
        and len(mlrun.mlconf.get_valid_function_priority_class_names())
    ):
        nuclio_spec.set_config(
            "spec.priorityClassName", function.spec.priority_class_name
        )

    if function.spec.service_account:
        nuclio_spec.set_config("spec.serviceAccount", function.spec.service_account)

    if function.spec.security_context:
        nuclio_spec.set_config(
            "spec.securityContext",
            mlrun.runtimes.pod.get_sanitized_attribute(
                function.spec, "security_context"
            ),
        )
    if function.spec.disable_default_http_trigger is not None:
        nuclio_spec.set_config(
            "spec.disableDefaultHTTPTrigger", function.spec.disable_default_http_trigger
        )


def _set_source_code_and_handler(function, config):
    if not function.spec.build.source:
        # set the source to the mlrun serving wrapper
        body = nuclio.build.mlrun_footer.format(
            mlrun.runtimes.nuclio.serving.serving_subkind
        )
        mlrun.utils.update_in(
            config,
            "spec.build.functionSourceCode",
            base64.b64encode(body.encode("utf-8")).decode("utf-8"),
        )
    elif not function.spec.function_handler:
        # point the nuclio function handler to mlrun serving wrapper handlers
        mlrun.utils.update_in(
            config,
            "spec.handler",
            "mlrun.serving.serving_wrapper:handler",
        )


@pure_nuclio_deployed_restricted()
def _resolve_and_set_base_image(
    function, config, client_version, client_python_version
):
    base_image = (
        mlrun.utils.get_in(config, "spec.build.baseImage")
        or function.spec.image
        or function.spec.build.base_image
    )
    if base_image:
        base_image = server.api.utils.builder.resolve_and_enrich_image_target(
            base_image,
            client_version=client_version,
            client_python_version=client_python_version,
        )
        mlrun.utils.update_in(
            config,
            "spec.build.baseImage",
            base_image,
        )


def _set_function_name(function, config, project, tag):
    name = mlrun.runtimes.nuclio.function.get_fullname(
        function.metadata.name, project, tag
    )
    function.status.nuclio_name = name
    mlrun.utils.update_in(config, "metadata.name", name)
    return name


def _add_secrets_config_to_function_spec(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
):
    handler = server.api.runtime_handlers.BaseRuntimeHandler
    if function.kind in [
        mlrun.runtimes.RuntimeKinds.remote,
        mlrun.runtimes.RuntimeKinds.nuclio,
        mlrun.runtimes.RuntimeKinds.application,
    ]:
        # For nuclio functions, we just add the project secrets as env variables. Since there's no MLRun code
        # to decode the secrets and special env variable names in the function, we just use the same env variable as
        # the key name (encode_key_names=False)
        handler.add_k8s_secrets_to_spec(
            None,
            function,
            project_name=function.metadata.project,
            encode_key_names=False,
        )

    elif function.kind == mlrun.runtimes.RuntimeKinds.serving:
        function: mlrun.runtimes.nuclio.serving.ServingRuntime
        if function.spec.secret_sources:
            function._secrets = mlrun.secrets.SecretsStore.from_list(
                function.spec.secret_sources
            )
            if function._secrets.has_vault_source():
                handler.add_vault_params_to_spec(
                    function, project_name=function.metadata.project
                )
            if function._secrets.has_azure_vault_source():
                handler.add_azure_vault_params_to_spec(
                    function, function._secrets.get_azure_vault_k8s_secret()
                )
            handler.add_k8s_secrets_to_spec(
                function._secrets.get_k8s_secrets(),
                function,
                project_name=function.metadata.project,
            )
        else:
            handler.add_k8s_secrets_to_spec(
                None, function, project_name=function.metadata.project
            )

    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Unexpected function kind {function.kind}. Expected one of: "
            f"{mlrun.runtimes.RuntimeKinds.nuclio_runtimes()}"
        )
