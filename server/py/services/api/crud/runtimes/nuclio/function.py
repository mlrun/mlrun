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

import asyncio
import base64
import os
import shlex

import nuclio
import nuclio.utils
import requests

import mlrun
import mlrun.auth.nuclio
import mlrun.common.constants
import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.datastore
import mlrun.errors
import mlrun.runtimes.nuclio.function
import mlrun.runtimes.pod
import mlrun.utils
from mlrun.k8s_utils import enrich_preemption_mode
from mlrun.utils import logger

import framework.utils.clients.async_nuclio
import framework.utils.clients.iguazio.v3
import framework.utils.singletons.k8s
import services.api.crud.runtimes.nuclio.helpers
import services.api.runtime_handlers
import services.api.utils.builder
from services.api.crud.runtimes.nuclio.helpers import pure_nuclio_deployed_restricted


def deploy_nuclio_function(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
    auth_info: mlrun.common.schemas.AuthInfo = None,
    client_version: str | None = None,
    builder_env: dict | None = None,
    client_python_version: str | None = None,
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
    services.api.crud.runtimes.nuclio.helpers.enrich_function_with_ingress(
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
            auth_info=mlrun.auth.nuclio.NuclioAuthInfo.from_auth_info(auth_info),
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
            auth_info=mlrun.auth.nuclio.NuclioAuthInfo.from_auth_info(auth_info),
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


async def delete_nuclio_functions_in_batches(
    auth_info: mlrun.common.schemas.AuthInfo,
    project_name: str,
    function_names: list[str],
):
    async def delete_function(
        nuclio_client: framework.utils.clients.iguazio.v3.AsyncClient,
        project: str,
        function: str,
        _semaphore: asyncio.Semaphore,
        k8s_helper_: framework.utils.singletons.k8s.K8sHelper,
    ) -> tuple[str, str] | None:
        async with _semaphore:
            try:
                await nuclio_client.delete_function(name=function, project_name=project)

                config_map = k8s_helper_.get_configmap(function)
                if config_map:
                    k8s_helper_.delete_configmap(config_map.metadata.name)
                return None
            except Exception as exc:
                # return tuple with failure info (intentionally not using mlrun.errors.err_to_str to avoid bloating
                # the failure message)
                return function, str(exc)

    # Configure maximum concurrent deletions
    max_concurrent_deletions = (
        mlrun.mlconf.background_tasks.function_deletion_batch_size
    )
    semaphore = asyncio.Semaphore(max_concurrent_deletions)
    failed_requests = []

    async with framework.utils.clients.async_nuclio.Client(auth_info) as client:
        k8s_helper = framework.utils.singletons.k8s.get_k8s_helper()
        tasks = [
            delete_function(client, project_name, function_name, semaphore, k8s_helper)
            for function_name in function_names
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # process results to identify failed deletion requests
        for result in results:
            if isinstance(result, tuple):
                nuclio_name, error_message = result
                if error_message:
                    failed_requests.append(error_message)

    return failed_requests


def _compile_function_config(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
    client_version: str | None = None,
    client_python_version: str | None = None,
    builder_env=None,
    auth_info=None,
):
    """
    Compile the nuclio function configuration from the mlrun function object.

    :param function:              mlrun function object
    :param client_version:        mlrun client version
    :param client_python_version: mlrun client python version
    :param builder_env:           mlrun builder environment (for config/credentials)
    :param auth_info:             service AuthInfo

    :return: function name, project name, nuclio function config
    """
    _enrich_config_spec(function, auth_info=auth_info)
    # resolve env vars before compiling the nuclio spec, as we need to set them in the spec
    env_dict, external_source_env_dict = _resolve_env_vars(function)

    project = function.metadata.project
    tag = function.metadata.tag

    # resolve sidecars images
    sidecars = function.spec.config.get("spec.sidecars") or []
    for sidecar in sidecars:
        sidecar_image = sidecar.get("image")
        if sidecar_image:
            sidecar["image"] = (
                services.api.utils.builder.resolve_and_enrich_image_target(
                    sidecar_image,
                    client_version=client_version,
                    client_python_version=client_python_version,
                )
            )

    # Application runtime always requires a sidecar (the user's app container).
    if function.kind == mlrun.runtimes.RuntimeKinds.application and not sidecars:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"No sidecar found for Application runtime '{function.metadata.name}'. "
            "Application runtime requires a sidecar container to run the user's application. "
            "Ensure the application image is set via 'spec.image' or 'with_sidecar()'."
        )

    # Configure init container when source needs runtime loading.
    if _should_fetch_source_code(function):
        if function.kind == mlrun.runtimes.RuntimeKinds.application:
            # Gate the project-secret envFrom mount on source type, not runtime
            # kind: store:// sources may resolve through DataStore profiles
            # whose private members live in mlrun-project-secrets-<project>;
            # git/archive sources don't need that resolution.
            application_source = function.spec.build.source or getattr(
                function.status, "application_source", ""
            )
            _configure_source_loader_init_container(
                function,
                # Application runtime has exactly one sidecar (the user's application container)
                sidecar=sidecars[0],
                client_version=client_version,
                client_python_version=client_python_version,
                add_project_secrets=mlrun.datastore.is_store_uri(application_source),
            )
        else:
            # Vanilla Nuclio/Serving. Nuclio's native builder resolves git://
            # and archive sources directly (codeEntryType="git"/"archive");
            # only store:// is unsupported and gets routed through the init
            # container path. Anything else with load_source_on_run=True is
            # an unsupported combination and raises.
            source = function.spec.build.source or getattr(
                function.status, "application_source", ""
            )
            if mlrun.datastore.is_store_uri(source):
                _install_store_uri_loader(
                    function,
                    project=project,
                    client_version=client_version,
                    client_python_version=client_python_version,
                )
            else:
                # _should_fetch_source_code returned True, source is not
                # store:// — must be git/archive + load_source_on_run on a
                # non-Application kind. The init-container redirect was only
                # designed for store://; other source types belong on
                # Application kind (where the sidecar handles them).
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Function '{function.metadata.name}' (kind="
                    f"{function.kind!r}) has source={source!r} with "
                    "load_source_on_run=True, which is supported on "
                    "Application kind only. Either set kind='application' "
                    "or use a store:// CodeArtifact source."
                )

    nuclio_spec = nuclio.ConfigSpec(
        env=env_dict,
        external_source_env=external_source_env_dict,
        config=function.spec.config,
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

        _set_function_metadata(function, config)

        if not mlrun.utils.get_in(config, "spec.handler"):
            # if handler was not set, we set it to the default value
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
        _set_function_metadata(function, config)

    mlrun.utils.update_in(
        config,
        "spec.volumes",
        function.spec.generate_nuclio_volumes(),
    )
    _resolve_and_set_base_image(function, config, client_version, client_python_version)
    _resolve_and_set_nuclio_runtime(
        function, config, client_version, client_python_version
    )
    _resolve_and_set_build_requirements_and_commands(function, config)

    function_name = _set_function_name(function, config, project, tag)

    return function_name, project, config


def _set_function_metadata(function, config):
    labels = function.metadata.labels or {}
    labels.update({mlrun_constants.MLRunInternalLabels.mlrun_class: function.kind})
    annotations = function.metadata.annotations or {}

    # make sure that labels and annotations exists in dictionary
    for key in [
        "metadata.labels",
        "metadata.annotations",
    ]:
        if not mlrun.utils.get_in(config, key):
            mlrun.utils.update_in(config, key, {})

    _apply_escaped_config(config, "metadata.labels", labels)
    _apply_escaped_config(config, "metadata.annotations", annotations)


def _apply_escaped_config(config, parent_key, items: dict):
    for key, value in items.items():
        # Adding escaping to the key to prevent it from being split by dots if it contains any
        mlrun.utils.update_in(config, f"{parent_key}.\\{key}\\", value)


def _enrich_config_spec(
    function, auth_info: mlrun.common.schemas.AuthInfo | None = None
):
    # Add secret configurations to function's pod spec, if secret sources were added.
    # Needs to be here, since it adds env params, which are handled in the next lines.
    # This only needs to run if we're running within k8s context. If running in Docker, for example, skip.
    if framework.utils.singletons.k8s.get_k8s_helper(
        silent=True
    ).is_running_inside_kubernetes_cluster():
        token_name = mlrun.utils.get_in(function.spec, "auth.token_name", None)
        _add_secrets_config_to_function_spec(function, token_name, auth_info)


def _resolve_env_vars(function):
    env_dict, external_source_env_dict = function._get_nuclio_config_spec_env()
    return env_dict, external_source_env_dict


def _resolve_and_set_nuclio_runtime(
    function, config, client_version, client_python_version
):
    nuclio_runtime = (
        function.spec.nuclio_runtime
        or services.api.crud.runtimes.nuclio.helpers.resolve_nuclio_runtime_python_image(
            mlrun_client_version=client_version, python_version=client_python_version
        )
        or mlrun.mlconf.default_nuclio_runtime
    )

    if mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
        "1.14.1",
    ) and nuclio_runtime in [
        "python:3.6",
        "python:3.7",
        "python:3.8",
    ]:
        nuclio_runtime_set_from_spec = nuclio_runtime == function.spec.nuclio_runtime
        if nuclio_runtime_set_from_spec and not mlrun.utils.get_in(
            config, "spec.build.baseImage"
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Nuclio version does not support the configured runtime: {nuclio_runtime}, and no base image was given"
            )
        else:
            logger.info(
                "Nuclio version does not support the configured runtime, using the default runtime",
                nuclio_runtime=nuclio_runtime,
                client_version=client_version,
                default_nuclio_runtime=mlrun.mlconf.default_nuclio_runtime,
            )
            nuclio_runtime = mlrun.mlconf.default_nuclio_runtime

    mlrun.utils.update_in(config, "spec.runtime", nuclio_runtime)


@pure_nuclio_deployed_restricted()
def _resolve_and_set_build_requirements_and_commands(function, config):
    _add_mlrun_to_requirements_if_needed(config, function)

    commands = (
        mlrun.utils.get_in(config, "spec.build.commands")
        or function.spec.build.commands
        or []
    )
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
        commands.append(f"python -m pip install {encoded_requirements}")

    mlrun.utils.update_in(config, "spec.build.commands", commands)


def _resolve_node_selector(run_db, project_name, function_node_selector):
    project_node_selector = {}

    if run_db and project_name:
        if project := run_db.get_project(project_name):
            project_node_selector = project.spec.default_function_node_selector

    return mlrun.runtimes.utils.resolve_node_selectors(
        project_node_selector, function_node_selector
    )


def _add_mlrun_to_requirements_if_needed(config, function):
    build: mlrun.model.ImageBuilder = function.spec.build
    base_image = mlrun.utils.get_in(config, "spec.build.baseImage")
    if (
        base_image
        and services.api.utils.builder.is_mlrun_image(base_image)
        and build.requirements
    ):
        services.api.utils.builder.add_mlrun_to_requirements(build, base_image)


def _set_build_params(function, nuclio_spec, builder_env, project, auth_info=None):
    # handle archive build params
    if function.spec.build.source:
        services.api.crud.runtimes.nuclio.helpers.compile_nuclio_archive_config(
            function, nuclio_spec, builder_env, project, auth_info=auth_info
        )

    if function.spec.no_cache:
        nuclio_spec.set_config("spec.build.noCache", True)
    if function.spec.build.functionSourceCode:
        nuclio_spec.set_config(
            "spec.build.functionSourceCode", function.spec.build.functionSourceCode
        )

    image_pull_secret = (
        services.api.crud.runtimes.nuclio.helpers.resolve_function_image_pull_secret(
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
            services.api.crud.runtimes.nuclio.helpers.parse_extra_args_to_nuclio_build_flags(
                function.spec.build.extra_args
            ),
        )


def _set_function_scheduling_params(function, nuclio_spec):
    node_selector = _resolve_node_selector(
        function._get_db(), function.metadata.project, function.spec.node_selector
    )
    affinity = function.spec.affinity
    tolerations = function.spec.tolerations
    preemption_mode = function.spec.preemption_mode

    # Enrich using preemption mode if defined
    (
        enriched_node_selector,
        enriched_tolerations,
        enriched_affinity,
    ) = enrich_preemption_mode(
        preemption_mode=preemption_mode,
        node_selector=node_selector,
        affinity=affinity,
        tolerations=tolerations,
    )

    if mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
        "1.5.20", "1.6.10"
    ):
        # We handle the enrichment of node selectors directly within MLRun, on the nuclio spec config.
        # This approach ensures that node selector settings from both the project and MLRun service levels
        # are incorporated into the Nuclio config.

        if enriched_node_selector:
            nuclio_spec.set_config("spec.nodeSelector", enriched_node_selector)

        if function.spec.node_name:
            nuclio_spec.set_config("spec.nodeName", function.spec.node_name)

        if enriched_affinity:
            nuclio_spec.set_config(
                "spec.affinity",
                mlrun.runtimes.pod.sanitize_attribute(enriched_affinity),
            )

    # don't send tolerations if nuclio is not compatible
    if mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility("1.7.5"):
        if enriched_tolerations:
            nuclio_spec.set_config(
                "spec.tolerations",
                mlrun.runtimes.pod.sanitize_attribute(enriched_tolerations),
            )

    if mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility("1.8.6"):
        if preemption_mode:
            nuclio_spec.set_config("spec.PreemptionMode", preemption_mode)


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

    if function.spec.custom_scaling_metric_specs:
        nuclio_spec.set_config(
            "spec.customScalingMetricSpecs",
            function.spec.custom_scaling_metric_specs,
        )

    # Nuclio supports spec.envFrom (mount all keys from secrets/configmaps)
    if function.spec.env_from:
        nuclio_spec.set_config(
            "spec.envFrom",
            mlrun.runtimes.pod.sanitize_attribute(function.spec.env_from),
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
        base_image = services.api.utils.builder.resolve_and_enrich_image_target(
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
    token_name: str,
    auth_info: mlrun.common.schemas.AuthInfo | None = None,
):
    handler = services.api.runtime_handlers.BaseRuntimeHandler
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
            token_name=token_name,
            auth_info=auth_info,
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
                token_name=token_name,
                auth_info=auth_info,
            )
        else:
            handler.add_k8s_secrets_to_spec(
                None,
                function,
                project_name=function.metadata.project,
                token_name=token_name,
                auth_info=auth_info,
            )

    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Unexpected function kind {function.kind}. Expected one of: "
            f"{mlrun.runtimes.RuntimeKinds.nuclio_runtimes()}"
        )


def _should_fetch_source_code(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
) -> bool:
    """
    Determine if an init container is needed for source loading.

    Init container is needed when:
    - Source is a store artifact URI (store://)
    - Source is Git or archive with pull_at_runtime=True

    Reads ``function.spec.build.source`` first, then falls back to
    ``function.status.application_source``. The fallback covers the redeploy
    case where ``_install_store_uri_loader`` cleared ``spec.build.source``
    and stashed the original URI on the status field so Nuclio's builder
    wouldn't see a URI it can't resolve.

    :param function: The function object
    :return: True if init container is needed, False otherwise
    """
    # build.source may be empty after from_image() clears it on redeploy.
    # fall back to status.application_source which preserves the original source URI.
    source = function.spec.build.source or getattr(
        function.status, "application_source", None
    )
    if not source:
        return False

    # Store artifact URIs always need init container
    if mlrun.datastore.is_store_uri(source):
        return True

    is_git_source = source.startswith("git://")
    is_archive_source = source.endswith(".tar.gz") or source.endswith(".zip")
    pull_at_runtime = function.spec.build.load_source_on_run

    return (is_git_source or is_archive_source) and pull_at_runtime


def _install_store_uri_loader(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
    project: str,
    client_version: str | None = None,
    client_python_version: str | None = None,
):
    """Install the store:// loader stub on a vanilla Nuclio/Serving function.

    For non-Application Nuclio/Serving kinds with a ``store://`` source,
    Nuclio's native builder cannot resolve the URI. This helper:

    1. Resolves the store URI and validates that the artifact is a
       CodeArtifact (``kind == "code"``); fails fast on a wrong-kind
       artifact rather than producing a cryptic ImportError in the pod.
    2. Stashes the URI to ``status.application_source`` and clears
       ``spec.build.source`` so Nuclio's builder doesn't see it.
    3. Bakes a generated loader stub as ``functionSourceCode`` under a
       fixed module name (avoids the ``sys.path[0]`` (= ``/opt/nuclio``)
       shadow over the user's real module).
    4. Stashes the user's real handler in ``status.original_handler`` and
       injects ``MLRUN_REAL_HANDLER`` into ``spec.config["spec.env"]``
       (deduped in both ``spec.config`` and ``spec.base_spec``).
    5. Overrides ``spec.function_handler`` to point at the loader.
    6. Configures the source-loader init container.

    Mutates ``function.spec`` and ``function.status``. Idempotent — safe to
    call on redeploys, which re-use the stashed values.

    :param function: The function to install the loader on.
    :param project:  Project name for artifact resolution.
    :param client_version: Forwarded to the init container builder.
    :param client_python_version: Forwarded to the init container builder.
    """
    source = function.spec.build.source or getattr(
        function.status, "application_source", ""
    )
    # Defensive guard: production callers (`_compile_function_config`) already
    # gate on `is_store_uri(source)`, but keep the no-op return so the helper
    # is safe to call from tests / future entry points without re-checking.
    if not mlrun.datastore.is_store_uri(source):
        return

    # Boundary validation: confirm the resolved artifact is a CodeArtifact.
    # Without this, a wrong-kind artifact (model, dataset) silently flows
    # through to the loader and fails with a cryptic ImportError at runtime.
    try:
        artifact = mlrun.datastore.get_store_resource(source, project=project)
    except mlrun.errors.MLRunBaseError:
        # Preserve typed MLRun errors so callers (and HTTP-status mapping)
        # can pattern-match on them; only the generic-Exception branch wraps.
        raise
    except Exception as exc:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Cannot resolve code artifact {source}: {mlrun.errors.err_to_str(exc)}"
        ) from exc
    # `get_store_resource` returns None when the resolver can't find the
    # artifact (deleted between deploys, empty resource, ...). Surface that
    # as a typed MLRunNotFoundError instead of the AttributeError that the
    # next `artifact.kind` access would otherwise raise.
    if artifact is None:
        raise mlrun.errors.MLRunNotFoundError(f"Code artifact not found at {source}")
    if artifact.kind != "code":
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Source {source} resolves to a {artifact.kind!r} artifact; "
            "expected a code artifact (kind='code')."
        )

    # Stash the URI so subsequent (re)deploys can re-resolve it after
    # spec.build.source is cleared for the Nuclio builder.
    if function.spec.build.source:
        function.status.application_source = function.spec.build.source
        function.spec.build.source = ""
    # base_spec must be set so _compile_function_config takes the config
    # branch; without it an empty source falls through to nuclio.build_file().
    if not function.spec.base_spec:
        function.spec.base_spec = nuclio.config.new_config()

    target_dir = (
        function.spec.build.source_code_target_dir
        or mlrun.common.constants.DEFAULT_SOURCE_CODE_TARGET_DIR
    )

    # Serving with no explicit handler uses mlrun's serving wrapper (wired
    # later by `_set_source_code_and_handler`). The loader stub would
    # override that and break the topology graph. Install the init container
    # + PYTHONPATH only; let `_set_source_code_and_handler` set the wrapper
    # handler. Serving WITH an explicit handler still goes through the
    # loader-stub path, same as nuclio.
    is_serving_topology = (
        function.kind == mlrun.runtimes.RuntimeKinds.serving
        and not function.spec.function_handler
    )
    loader_module = mlrun.common.constants.STORE_URI_HANDLER_LOADER_MODULE
    loader_handler = f"{loader_module}:handler"

    if is_serving_topology:
        # If the function was previously deployed with an explicit handler
        # (loader stub baked, MLRUN_REAL_HANDLER env injected) and the user
        # is now switching to topology, clear the loader-shaped state so
        # the spec doesn't carry dead bytes. The serving wrapper handler
        # is wired downstream by `_set_source_code_and_handler`.
        if function.spec.function_handler == loader_handler:
            function.spec.function_handler = ""
        env_list = function.spec.config.get("spec.env") or []
        env_list[:] = [e for e in env_list if e.get("name") != "MLRUN_REAL_HANDLER"]
        # Same dedupe in base_spec as `_set_unique_env` does for the
        # explicit-handler path.
        base_spec = function.spec.base_spec or {}
        base_env = mlrun.utils.get_in(base_spec, "spec.env", []) or []
        if base_env:
            mlrun.utils.update_in(
                base_spec,
                "spec.env",
                [e for e in base_env if e.get("name") != "MLRUN_REAL_HANDLER"],
            )
        # If the previously-baked stub is loader-shaped, drop it so Nuclio
        # rebuilds with the wrapper wiring.
        if function.spec.build.functionSourceCode:
            try:
                decoded = base64.b64decode(
                    function.spec.build.functionSourceCode
                ).decode("utf-8")
            except Exception:
                decoded = ""
            if "MLRUN_REAL_HANDLER" in decoded:
                function.spec.build.functionSourceCode = ""

        logger.debug(
            "Installing source-loader init container for serving topology",
            project=function.metadata.project,
            function=function.metadata.name,
            source=source,
        )
        _configure_source_loader_init_container(
            function,
            sidecar=None,
            client_version=client_version,
            client_python_version=client_python_version,
            add_project_secrets=True,
        )
        return

    # Stub-bytes layout: a version marker on the first line lets future
    # mlrun releases re-bake an existing function's loader when stub
    # semantics change. Without the marker, the `if not functionSourceCode`
    # short-circuit below would keep stale bytes from the first deploy
    # forever. Marker version lives in `mlrun.common.constants` —
    # bump it whenever the stub body changes.
    stub_version_marker = (
        f"# stub_version={mlrun.common.constants.STORE_URI_LOADER_STUB_VERSION}"
    )
    existing_stub = ""
    if function.spec.build.functionSourceCode:
        try:
            existing_stub = base64.b64decode(
                function.spec.build.functionSourceCode
            ).decode("utf-8")
        except Exception:
            existing_stub = ""
    # Substring check (not splitlines) so the marker is robust to line-ending
    # mutations if the bytes were ever round-tripped through a Windows tool.
    if stub_version_marker not in existing_stub:
        loader_source = (
            f"{stub_version_marker}\n"
            "# Generated by mlrun for store:// Nuclio functions.\n"
            "# Loads the user's real handler from the source-loader\n"
            "# init container's output directory.\n"
            "import importlib\n"
            "import os\n"
            "import sys\n"
            "\n"
            f"_CODE_DIR = {target_dir!r}\n"
            "if _CODE_DIR not in sys.path:\n"
            "    sys.path.insert(0, _CODE_DIR)\n"
            "\n"
            '_real = os.environ.get("MLRUN_REAL_HANDLER", "")\n'
            "if not _real:\n"
            "    raise RuntimeError(\n"
            '        "MLRUN_REAL_HANDLER env var not set; cannot resolve "\n'
            '        "user handler for store:// Nuclio function."\n'
            "    )\n"
            '_module_name, _, _func_name = _real.rpartition(":")\n'
            "if not _module_name or not _func_name:\n"
            "    raise RuntimeError(\n"
            "        f\"MLRUN_REAL_HANDLER must be in 'module:function' \"\n"
            '        f"format, got {_real!r}."\n'
            "    )\n"
            "\n"
            "# Drop any cached _module_name from a prior pod-warm-start so a\n"
            "# redeploy that re-uses the worker picks up the new bytes the\n"
            "# init container fetched into _CODE_DIR.\n"
            "importlib.invalidate_caches()\n"
            "if _module_name in sys.modules:\n"
            "    del sys.modules[_module_name]\n"
            "\n"
            "_user_module = importlib.import_module(_module_name)\n"
            "handler = getattr(_user_module, _func_name)\n"
        )
        function.spec.build.functionSourceCode = base64.b64encode(
            loader_source.encode("utf-8")
        ).decode("utf-8")

    # Resolve the user's "real" handler. The spec is the source of truth:
    # if the user updates `function.spec.function_handler` between deploys,
    # the new value wins. status.original_handler is only consulted as a
    # fallback for the redeploy case where spec.function_handler has
    # already been overwritten with the loader handler from a prior deploy.
    # When neither is set (first deploy with no handler= passed), fall back
    # to mlrun's existing convention default — the same `"main:handler"`
    # used elsewhere in this file for non-store:// nuclio functions.
    spec_handler = function.spec.function_handler
    if spec_handler and spec_handler != loader_handler:
        # User-provided value (first deploy or explicit redeploy update).
        function.status.original_handler = spec_handler
    elif not function.status.original_handler:
        function.status.original_handler = "main:handler"

    _set_unique_env(function, "MLRUN_REAL_HANDLER", function.status.original_handler)

    # Override function_handler so Nuclio's CRD spec.handler points at the
    # loader.
    function.spec.function_handler = loader_handler

    logger.debug(
        "Installed store:// loader stub on Nuclio function",
        project=function.metadata.project,
        function=function.metadata.name,
        source=source,
        original_handler=function.status.original_handler,
    )

    _configure_source_loader_init_container(
        function,
        sidecar=None,
        client_version=client_version,
        client_python_version=client_python_version,
        add_project_secrets=True,
    )


def _set_unique_env(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
    name: str,
    value: str,
):
    """Set an env var on the function, deduping in both ``spec.config`` and
    ``spec.base_spec``.

    K8s tolerates duplicate env names (last wins), but
    ``nuclio.config.extend_config`` later appends ``spec.config["spec.env"]``
    onto whatever's in ``spec.base_spec["spec"]["env"]``, which on a
    redeploy can produce two entries for the same name. Mirror the dedupe
    pattern used for init containers: clear from both, then append fresh
    to ``spec.config`` only.
    """
    env_list = function.spec.config.setdefault("spec.env", [])
    env_list[:] = [e for e in env_list if e.get("name") != name]
    env_list.append({"name": name, "value": value})

    base_spec = function.spec.base_spec or {}
    base_env = mlrun.utils.get_in(base_spec, "spec.env", []) or []
    if base_env:
        mlrun.utils.update_in(
            base_spec,
            "spec.env",
            [e for e in base_env if e.get("name") != name],
        )


def _configure_source_loader_init_container(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
    sidecar: dict | None = None,
    client_version: str | None = None,
    client_python_version: str | None = None,
    add_project_secrets: bool = False,
):
    """
    Configure an init container to load source code at runtime.

    This function sets up a Kubernetes init container that runs before the main
    container starts. The init container is responsible for fetching source code
    from remote locations (store:// URIs, git repos, archives) and extracting it
    to a shared volume that the runtime container can access.

    For Application runtime the runtime container is the user-provided sidecar
    (passed in ``sidecar``). For vanilla Nuclio/Serving there is no sidecar, and
    the runtime container is the main function container itself — pass
    ``sidecar=None`` and the main container is patched via ``function.spec.config``.

    The setup involves:
    1. Creating an emptyDir volume shared between init container and runtime container
    2. Building an init container spec that runs `mlrun load-source` command
    3. Adding the init container to the function's Nuclio spec
    4. Patching the runtime container to mount the shared volume and set PYTHONPATH

    :param function: The function object to configure
    :param sidecar: The sidecar container dict (the user's application container)
                    when patching an Application sidecar; ``None`` to patch the
                    main Nuclio function container instead.
    :param client_version: Client version for resolving the init container image
    :param client_python_version: Client Python version for resolving the init container image
    :param add_project_secrets: Forwarded to _build_source_loader_init_container.
                                When True, mounts the project's K8s secret via
                                envFrom on the init container so the loader can
                                authenticate to credential-protected datastores
                                and resolve DataStore profiles with private
                                members. Off for non-store:// sources, whose
                                loaders read no project secrets.
    """
    source = function.spec.build.source or getattr(
        function.status, "application_source", None
    )
    workdir = function.spec.workdir
    target_dir = (
        function.spec.build.source_code_target_dir
        or mlrun.common.constants.DEFAULT_SOURCE_CODE_TARGET_DIR
    )

    # Create shared volume for source code
    volume_name = mlrun.common.constants.SOURCE_CODE_VOLUME_NAME
    volume = {"name": volume_name, "emptyDir": {}}
    volume_mount = {"name": volume_name, "mountPath": target_dir}

    # Add volume to function spec so both init container and runtime container can access it
    function.spec.with_volumes(volume)
    function.spec.with_volume_mounts(volume_mount)

    # Build the init container spec with mlrun load-source command
    init_container = _build_source_loader_init_container(
        function=function,
        source=source,
        target_dir=target_dir,
        volume_mount=volume_mount,
        client_version=client_version,
        client_python_version=client_python_version,
        add_project_secrets=add_project_secrets,
    )

    # Add init container to function spec (idempotently - replaces if exists)
    _ensure_source_loader_init_container(function, init_container)

    if sidecar is not None:
        _patch_sidecar_for_source(
            sidecar=sidecar,
            volume_name=volume_name,
            volume_mount=volume_mount,
            target_dir=target_dir,
            workdir=workdir,
        )
    else:
        # Vanilla Nuclio/serving: inject PYTHONPATH on the main function
        # container so the runtime can import the loaded source. Volume
        # mounts are already wired via function.spec.with_volume_mounts()
        # above.
        _inject_main_container_pythonpath(
            function=function,
            target_dir=target_dir,
            workdir=workdir,
        )

    logger.debug(
        "Configured source loader init container",
        project=function.metadata.project,
        function=function.metadata.name,
        source=source,
        target_dir=target_dir,
        workdir=function.spec.workdir,
    )


def _build_source_loader_init_container(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
    source: str,
    target_dir: str,
    volume_mount: dict,
    client_version: str | None = None,
    client_python_version: str | None = None,
    add_project_secrets: bool = False,
) -> dict:
    """
    Build the init container spec for loading source code.

    :param function: The function object
    :param source: Source URI to load
    :param target_dir: Target directory for source code
    :param volume_mount: Volume mount configuration
    :param client_version: Client version for image resolution
    :param client_python_version: Client Python version for image resolution
    :param add_project_secrets: When True, mount the project's K8s secret
                                via envFrom so the loader process can use
                                project-secret-based credentials (e.g., AWS_*,
                                GOOGLE_APPLICATION_CREDENTIALS) AND resolve
                                DataStore profiles with private members
                                (which are stored in the same K8s secret).
                                Off for non-store:// sources, where the
                                loader has no datastore-profile resolution
                                to do.
    :return: Init container specification dict
    """
    project = function.metadata.project

    init_container_image = services.api.utils.builder.resolve_and_enrich_image_target(
        mlrun.mlconf.default_base_image,
        client_version=client_version,
        client_python_version=client_python_version,
    )

    init_container = {
        "name": mlrun.common.constants.SOURCE_LOADER_INIT_CONTAINER_NAME,
        "image": init_container_image,
        "command": ["mlrun", "load-source"],
        "args": [source, "--project", project, "--target", target_dir],
        "env": [
            {"name": "MLRUN_PROJECT", "value": project},
            {"name": "MLRUN_DBPATH", "value": mlrun.mlconf.httpdb.api_url},
        ],
        "volumeMounts": [volume_mount],
    }

    if add_project_secrets:
        # `optional: true` keeps deploys working when a project has no secrets
        # set yet — set_secrets({}) creates the K8s Secret lazily.
        init_container["envFrom"] = [
            {
                "secretRef": {
                    "name": framework.utils.singletons.k8s.get_k8s_helper(
                        silent=True
                    ).get_project_secret_name(project),
                    "optional": True,
                }
            }
        ]

    return init_container


def _ensure_source_loader_init_container(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
    init_container: dict,
):
    """
    Add the source loader init container to the function spec idempotently.

    If an init container with the same name already exists, it is filtered
    out and replaced with the new configuration. The dedupe runs against
    BOTH ``function.spec.config["spec.initContainers"]`` AND the matching
    list inside ``function.spec.base_spec``: on a redeploy the persisted
    base_spec carries over the previous deploy's init container, and
    ``nuclio.config.extend_config`` later merges base_spec into config —
    without the base_spec dedupe, that merge produces a duplicate and
    Nuclio rejects the resulting K8s Deployment with "Duplicate value".

    :param function: The function object to configure
    :param init_container: Init container specification
    """
    init_container_name = init_container.get("name")
    if not init_container_name:
        raise mlrun.errors.MLRunInvalidArgumentError("Init container name is required")

    # Dedupe in function.spec.config first.
    init_containers = function.spec.config.setdefault("spec.initContainers", [])
    init_containers[:] = [
        c for c in init_containers if c.get("name") != init_container_name
    ]
    init_containers.append(init_container)

    # Also dedupe in function.spec.base_spec — on redeploys the persisted
    # base_spec carries over the previous deploy's init container at
    # spec.initContainers, and extend_config later merges base_spec with
    # function.spec.config, producing a duplicate. Nuclio rejects the
    # resulting K8s Deployment with "Duplicate value".
    base_spec = function.spec.base_spec or {}
    base_spec_init_containers = mlrun.utils.get_in(base_spec, "spec.initContainers", [])
    if base_spec_init_containers:
        mlrun.utils.update_in(
            base_spec,
            "spec.initContainers",
            [
                c
                for c in base_spec_init_containers
                if c.get("name") != init_container_name
            ],
        )


def _resolve_workdir(target_dir: str, workdir: str | None) -> str:
    """Resolve a workdir against the source-loader target dir.

    Absolute workdirs are used as-is; relative workdirs are joined onto
    target_dir. Falls back to target_dir when workdir is empty.
    """
    if not workdir:
        return target_dir
    return workdir if os.path.isabs(workdir) else os.path.join(target_dir, workdir)


def _add_pythonpath_to_env_list(env_list: list[dict], resolved_workdir: str):
    """Idempotently prepend resolved_workdir to a PYTHONPATH entry in env_list.

    If PYTHONPATH is missing, append a fresh entry. If present and already
    contains resolved_workdir (as a colon-separated entry), leave it alone.
    Otherwise prepend, preserving any user-defined paths.
    """
    pythonpath_env = next((e for e in env_list if e.get("name") == "PYTHONPATH"), None)
    if pythonpath_env is None:
        env_list.append({"name": "PYTHONPATH", "value": resolved_workdir})
        return
    existing_path = pythonpath_env.get("value", "")
    if resolved_workdir in existing_path.split(":"):
        return
    pythonpath_env["value"] = (
        f"{resolved_workdir}:{existing_path}" if existing_path else resolved_workdir
    )


def _patch_sidecar_for_source(
    sidecar: dict,
    volume_name: str,
    volume_mount: dict,
    target_dir: str,
    workdir: str | None = None,
):
    """
    Patch sidecar container with volume mount, workingDir, and PYTHONPATH.

    :param sidecar: The sidecar container dict
    :param volume_name: Name of the source volume
    :param volume_mount: Volume mount configuration
    :param target_dir: Target directory where source code is extracted
    :param workdir: Working directory relative to target_dir (e.g. 'subdir') or absolute path
                    on the container filesystem. When set, the sidecar runs from this directory
                    instead of the target_dir root.
    """
    # Add volume mount idempotently
    sidecar_mounts = sidecar.setdefault("volumeMounts", [])
    if not any(vm.get("name") == volume_name for vm in sidecar_mounts):
        sidecar_mounts.append(volume_mount)

    resolved_workdir = _resolve_workdir(target_dir, workdir)
    sidecar["workingDir"] = resolved_workdir

    _add_pythonpath_to_env_list(sidecar.setdefault("env", []), resolved_workdir)


def _inject_main_container_pythonpath(
    function: mlrun.runtimes.nuclio.function.RemoteRuntime,
    target_dir: str,
    workdir: str | None = None,
):
    """
    Inject PYTHONPATH into ``function.spec.config["spec.env"]`` so Nuclio
    applies it to the main function container at deploy time.

    Unlike sidecars (which are dicts directly mutable in
    ``function.spec.config["spec.sidecars"]``), the main container is
    constructed by Nuclio's controller from the function CRD's ``spec.env``.
    We append to that env list here because ``_resolve_env_vars()`` — which
    would otherwise have picked up ``function.spec.env`` — has already run
    by the time this function is called.

    Volume mounts for the main container are wired by the caller via
    ``function.spec.with_volume_mounts()``.

    :param function: The function object
    :param target_dir: Target directory where source code is extracted
    :param workdir: Working directory relative to target_dir or absolute path on
                    the container filesystem
    """
    _add_pythonpath_to_env_list(
        function.spec.config.setdefault("spec.env", []),
        _resolve_workdir(target_dir, workdir),
    )

    # Mirror the dedupe in spec.base_spec — extend_config merges
    # base_spec["spec"]["env"] into spec.config["spec.env"] on deploy, and
    # without removing the stale PYTHONPATH from base_spec a redeploy
    # accumulates a duplicate entry alongside the freshly-set one.
    base_spec = function.spec.base_spec or {}
    base_env = mlrun.utils.get_in(base_spec, "spec.env", []) or []
    if base_env:
        mlrun.utils.update_in(
            base_spec,
            "spec.env",
            [e for e in base_env if e.get("name") != "PYTHONPATH"],
        )
