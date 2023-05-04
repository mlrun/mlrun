# Copyright 2018 Iguazio
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
import urllib.parse

import nuclio
import nuclio.utils
import requests
import semver

import mlrun
import mlrun.api.schemas
import mlrun.api.utils.singletons.k8s
import mlrun.datastore
import mlrun.errors
import mlrun.runtimes.function
import mlrun.utils
from mlrun.utils import logger


def deploy_nuclio_function(
    function: mlrun.runtimes.function.RemoteRuntime,
    auth_info: mlrun.api.schemas.AuthInfo = None,
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
    _enrich_function_with_ingress(
        function_config,
        function.spec.add_templated_ingress_host_mode
        or mlrun.mlconf.httpdb.nuclio.add_templated_ingress_host_mode,
        function.spec.service_type or mlrun.mlconf.httpdb.nuclio.default_service_type,
    )

    try:
        return nuclio.deploy.deploy_config(
            function_config,
            dashboard_url=mlrun.mlconf.nuclio_dashboard_url,
            name=function_name,
            project=project_name,
            tag=function.metadata.tag,
            verbose=function.verbose,
            create_new=True,
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
    auth_info: mlrun.api.schemas.AuthInfo = None,
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
    name = mlrun.runtimes.function.get_fullname(name, project, tag)
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


def _compile_function_config(
    function: mlrun.runtimes.function.RemoteRuntime,
    client_version: str = None,
    client_python_version: str = None,
    builder_env=None,
    auth_info=None,
):
    labels = function.metadata.labels or {}
    labels.update({"mlrun/class": function.kind})
    for key, value in labels.items():
        # Adding escaping to the key to prevent it from being split by dots if it contains any
        function.set_config(f"metadata.labels.\\{key}\\", value)

    # Add secret configurations to function's pod spec, if secret sources were added.
    # Needs to be here, since it adds env params, which are handled in the next lines.
    # This only needs to run if we're running within k8s context. If running in Docker, for example, skip.
    if mlrun.api.utils.singletons.k8s.get_k8s_helper(
        silent=True
    ).is_running_inside_kubernetes_cluster():
        function.add_secrets_config_to_spec()

    env_dict, external_source_env_dict = function._get_nuclio_config_spec_env()

    nuclio_runtime = (
        function.spec.nuclio_runtime
        or _resolve_nuclio_runtime_python_image(
            mlrun_client_version=client_version, python_version=client_python_version
        )
    )

    if _is_nuclio_version_in_range("0.0.0", "1.6.0") and nuclio_runtime in [
        "python:3.7",
        "python:3.8",
    ]:
        nuclio_runtime_set_from_spec = nuclio_runtime == function.spec.nuclio_runtime
        if nuclio_runtime_set_from_spec:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Nuclio version does not support the configured runtime: {nuclio_runtime}"
            )
        else:
            # our default is python:3.9, simply set it to python:3.6 to keep supporting envs with old Nuclio
            nuclio_runtime = "python:3.6"

    # In nuclio 1.6.0<=v<1.8.0, python runtimes default behavior was to not decode event strings
    # Our code is counting on the strings to be decoded, so add the needed env var for those versions
    if (
        _is_nuclio_version_in_range("1.6.0", "1.8.0")
        and "NUCLIO_PYTHON_DECODE_EVENT_STRINGS" not in env_dict
    ):
        env_dict["NUCLIO_PYTHON_DECODE_EVENT_STRINGS"] = "true"

    nuclio_spec = nuclio.ConfigSpec(
        env=env_dict,
        external_source_env=external_source_env_dict,
        config=function.spec.config,
    )
    nuclio_spec.cmd = function.spec.build.commands or []

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

    project = function.metadata.project or "default"
    tag = function.metadata.tag
    handler = function.spec.function_handler

    if function.spec.build.source:
        _compile_nuclio_archive_config(
            nuclio_spec, function, builder_env, project, auth_info=auth_info
        )

    nuclio_spec.set_config("spec.runtime", nuclio_runtime)

    # In Nuclio >= 1.6.x default serviceType has changed to "ClusterIP".
    nuclio_spec.set_config(
        "spec.serviceType",
        function.spec.service_type or mlrun.mlconf.httpdb.nuclio.default_service_type,
    )
    if function.spec.readiness_timeout:
        nuclio_spec.set_config(
            "spec.readinessTimeoutSeconds", function.spec.readiness_timeout
        )
    if function.spec.resources:
        nuclio_spec.set_config("spec.resources", function.spec.resources)
    if function.spec.no_cache:
        nuclio_spec.set_config("spec.build.noCache", True)
    if function.spec.build.functionSourceCode:
        nuclio_spec.set_config(
            "spec.build.functionSourceCode", function.spec.build.functionSourceCode
        )

    image_pull_secret = _resolve_function_image_pull_secret(function)
    if image_pull_secret:
        nuclio_spec.set_config("spec.imagePullSecrets", image_pull_secret)

    if function.spec.base_image_pull:
        nuclio_spec.set_config("spec.build.noBaseImagesPull", False)
    # don't send node selections if nuclio is not compatible
    if mlrun.runtimes.function.validate_nuclio_version_compatibility(
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
    if mlrun.runtimes.function.validate_nuclio_version_compatibility("1.7.5"):
        if function.spec.tolerations:
            nuclio_spec.set_config(
                "spec.tolerations",
                mlrun.runtimes.pod.get_sanitized_attribute(
                    function.spec, "tolerations"
                ),
            )
    # don't send preemption_mode if nuclio is not compatible
    if mlrun.runtimes.function.validate_nuclio_version_compatibility("1.8.6"):
        if function.spec.preemption_mode:
            nuclio_spec.set_config(
                "spec.PreemptionMode",
                function.spec.preemption_mode,
            )

    # don't send default or any priority class name if nuclio is not compatible
    if (
        function.spec.priority_class_name
        and mlrun.runtimes.function.validate_nuclio_version_compatibility("1.6.18")
        and len(mlrun.mlconf.get_valid_function_priority_class_names())
    ):
        nuclio_spec.set_config(
            "spec.priorityClassName", function.spec.priority_class_name
        )

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

    if function.spec.service_account:
        nuclio_spec.set_config("spec.serviceAccount", function.spec.service_account)

    if function.spec.security_context:
        nuclio_spec.set_config(
            "spec.securityContext",
            mlrun.runtimes.pod.get_sanitized_attribute(
                function.spec, "security_context"
            ),
        )

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

        mlrun.utils.update_in(config, "metadata.name", function.metadata.name)
        mlrun.utils.update_in(
            config, "spec.volumes", function.spec.generate_nuclio_volumes()
        )
        base_image = (
            mlrun.utils.get_in(config, "spec.build.baseImage")
            or function.spec.image
            or function.spec.build.base_image
        )
        if base_image:
            mlrun.utils.update_in(
                config,
                "spec.build.baseImage",
                mlrun.utils.enrich_image_url(
                    base_image, client_version, client_python_version
                ),
            )

        logger.info("deploy started")
        name = mlrun.runtimes.function.get_fullname(
            function.metadata.name, project, tag
        )
        function.status.nuclio_name = name
        mlrun.utils.update_in(config, "metadata.name", name)

        if (
            function.kind == mlrun.runtimes.RuntimeKinds.serving
            and not mlrun.utils.get_in(config, "spec.build.functionSourceCode")
        ):
            if not function.spec.build.source:
                # set the source to the mlrun serving wrapper
                body = nuclio.build.mlrun_footer.format(
                    mlrun.runtimes.serving.serving_subkind
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
    else:
        # this may also be called in case of using single file code_to_function(embed_code=False)
        # this option need to be removed or be limited to using remote files (this code runs in server)
        name, config, code = nuclio.build_file(
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
        base_image = function.spec.image or function.spec.build.base_image
        if base_image:
            mlrun.utils.update_in(
                config,
                "spec.build.baseImage",
                mlrun.utils.enrich_image_url(
                    base_image, client_version, client_python_version
                ),
            )

        name = mlrun.runtimes.function.get_fullname(name, project, tag)
        function.status.nuclio_name = name

        mlrun.utils.update_in(config, "metadata.name", name)

    return name, project, config


def _enrich_function_with_ingress(config, mode, service_type):
    # do not enrich with an ingress
    if mode == mlrun.runtimes.constants.NuclioIngressAddTemplatedIngressModes.never:
        return

    ingresses = _resolve_function_ingresses(config["spec"])

    # function has ingresses already, nothing to add / enrich
    if ingresses:
        return

    # if exists, get the http trigger the function has
    # we would enrich it with an ingress
    http_trigger = resolve_function_http_trigger(config["spec"])
    if not http_trigger:
        # function has an HTTP trigger without an ingress
        # TODO: read from nuclio-api frontend-spec
        http_trigger = {
            "kind": "http",
            "name": "http",
            "maxWorkers": 1,
            "workerAvailabilityTimeoutMilliseconds": 10000,  # 10 seconds
            "attributes": {},
        }

    def enrich():
        http_trigger.setdefault("attributes", {}).setdefault("ingresses", {})["0"] = {
            "paths": ["/"],
            # this would tell Nuclio to use its default ingress host template
            # and would auto assign a host for the ingress
            "hostTemplate": "@nuclio.fromDefault",
        }
        http_trigger["attributes"]["serviceType"] = service_type
        config["spec"].setdefault("triggers", {})[http_trigger["name"]] = http_trigger

    if mode == mlrun.runtimes.constants.NuclioIngressAddTemplatedIngressModes.always:
        enrich()
    elif (
        mode
        == mlrun.runtimes.constants.NuclioIngressAddTemplatedIngressModes.on_cluster_ip
    ):

        # service type is not cluster ip, bail out
        if service_type and service_type.lower() != "clusterip":
            return

        enrich()


def _resolve_function_ingresses(function_spec):
    http_trigger = resolve_function_http_trigger(function_spec)
    if not http_trigger:
        return []

    ingresses = []
    for _, ingress_config in (
        http_trigger.get("attributes", {}).get("ingresses", {}).items()
    ):
        ingresses.append(ingress_config)
    return ingresses


def resolve_function_http_trigger(function_spec):
    for trigger_name, trigger_config in function_spec.get("triggers", {}).items():
        if trigger_config.get("kind") != "http":
            continue
        return trigger_config


def _resolve_nuclio_runtime_python_image(
    mlrun_client_version: str = None, python_version: str = None
):
    # if no python version or mlrun version is passed it means we use mlrun client older than 1.3.0 therefore need
    # to use the previoud default runtime which is python 3.7
    if not python_version or not mlrun_client_version:
        return "python:3.7"

    # If the mlrun version is 0.0.0-<unstable>, it is a dev version,
    # so we can't check if it is higher than 1.3.0, but if the python version was passed,
    # it means it is 1.3.0-rc or higher, so use the image according to the python version
    if mlrun_client_version.startswith("0.0.0-") or "unstable" in mlrun_client_version:
        if python_version.startswith("3.7"):
            return "python:3.7"

        return mlrun.mlconf.default_nuclio_runtime

    # if mlrun version is older than 1.3.0 we need to use the previous default runtime which is python 3.7
    if semver.VersionInfo.parse(mlrun_client_version) < semver.VersionInfo.parse(
        "1.3.0-X"
    ):
        return "python:3.7"

    # if mlrun version is 1.3.0 or newer and python version is 3.7 we need to use python 3.7 image
    if semver.VersionInfo.parse(mlrun_client_version) >= semver.VersionInfo.parse(
        "1.3.0-X"
    ) and python_version.startswith("3.7"):
        return "python:3.7"

    # if none of the above conditions are met we use the default runtime which is python 3.9
    return mlrun.mlconf.default_nuclio_runtime


def _compile_nuclio_archive_config(
    nuclio_spec,
    function: mlrun.runtimes.function.RemoteRuntime,
    builder_env,
    project=None,
    auth_info=None,
):
    secrets = {}
    if (
        project
        and mlrun.api.utils.singletons.k8s.get_k8s_helper().is_running_inside_kubernetes_cluster()
    ):
        secrets = (
            mlrun.api.utils.singletons.k8s.get_k8s_helper().get_project_secret_data(
                project
            )
        )

    def get_secret(key):
        return builder_env.get(key) or secrets.get(key, "")

    source = function.spec.build.source
    parsed_url = urllib.parse.urlparse(source)
    code_entry_type = ""
    if source.startswith("s3://"):
        code_entry_type = "s3"
    if source.startswith("git://"):
        code_entry_type = "git"
    for archive_prefix in ["http://", "https://", "v3io://", "v3ios://"]:
        if source.startswith(archive_prefix):
            code_entry_type = "archive"

    if code_entry_type == "":
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Couldn't resolve code entry type from source"
        )

    code_entry_attributes = {}

    # resolve work_dir and handler
    work_dir, handler = _resolve_work_dir_and_handler(function.spec.function_handler)
    work_dir = function.spec.workdir or work_dir
    if work_dir != "":
        code_entry_attributes["workDir"] = work_dir

    # archive
    if code_entry_type == "archive":
        v3io_access_key = builder_env.get("V3IO_ACCESS_KEY", "")
        if source.startswith("v3io"):
            if not parsed_url.netloc:
                source = mlrun.mlconf.v3io_api + parsed_url.path
            else:
                source = f"http{source[len('v3io'):]}"
            if auth_info and not v3io_access_key:
                v3io_access_key = auth_info.data_session or auth_info.access_key

        if v3io_access_key:
            code_entry_attributes["headers"] = {"X-V3io-Session-Key": v3io_access_key}

    # s3
    if code_entry_type == "s3":
        bucket, item_key = mlrun.datastore.parse_s3_bucket_and_key(source)

        code_entry_attributes["s3Bucket"] = bucket
        code_entry_attributes["s3ItemKey"] = item_key

        code_entry_attributes["s3AccessKeyId"] = get_secret("AWS_ACCESS_KEY_ID")
        code_entry_attributes["s3SecretAccessKey"] = get_secret("AWS_SECRET_ACCESS_KEY")
        code_entry_attributes["s3SessionToken"] = get_secret("AWS_SESSION_TOKEN")

    # git
    if code_entry_type == "git":

        # change git:// to https:// as nuclio expects it to be
        if source.startswith("git://"):
            source = source.replace("git://", "https://")

        source, reference, branch = mlrun.utils.resolve_git_reference_from_source(
            source
        )
        if not branch and not reference:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "git branch or refs must be specified in the source e.g.: "
                "'git://<url>/org/repo.git#<branch-name or refs/heads/..>'"
            )
        if reference:
            code_entry_attributes["reference"] = reference
        if branch:
            code_entry_attributes["branch"] = branch

        password = get_secret("GIT_PASSWORD")
        username = get_secret("GIT_USERNAME")

        token = get_secret("GIT_TOKEN")
        if token:
            username, password = mlrun.utils.get_git_username_password_from_token(token)

        code_entry_attributes["username"] = username
        code_entry_attributes["password"] = password

    # populate spec with relevant fields
    nuclio_spec.set_config("spec.handler", handler)
    nuclio_spec.set_config("spec.build.path", source)
    nuclio_spec.set_config("spec.build.codeEntryType", code_entry_type)
    nuclio_spec.set_config("spec.build.codeEntryAttributes", code_entry_attributes)


def _resolve_function_image_pull_secret(function):
    """
    the corresponding attribute for 'build.secret' in nuclio is imagePullSecrets, attached link for reference
    https://github.com/nuclio/nuclio/blob/e4af2a000dc52ee17337e75181ecb2652b9bf4e5/pkg/processor/build/builder.go#L1073
    if only one of the secrets is set, use it.
    if both are set, use the non default one and give precedence to image_pull_secret
    """
    # enrich only on server side
    if not mlrun.config.is_running_as_api():
        return function.spec.image_pull_secret or function.spec.build.secret

    if function.spec.image_pull_secret is None:
        function.spec.image_pull_secret = (
            mlrun.mlconf.function.spec.image_pull_secret.default
        )
    elif (
        function.spec.image_pull_secret
        != mlrun.mlconf.function.spec.image_pull_secret.default
    ):
        return function.spec.image_pull_secret

    if function.spec.build.secret is None:
        function.spec.build.secret = mlrun.mlconf.httpdb.builder.docker_registry_secret
    elif (
        function.spec.build.secret != mlrun.mlconf.httpdb.builder.docker_registry_secret
    ):
        return function.spec.build.secret

    return function.spec.image_pull_secret or function.spec.build.secret


def _resolve_work_dir_and_handler(handler):
    """
    Resolves a nuclio function working dir and handler inside an archive/git repo
    :param handler: a path describing working dir and handler of a nuclio function
    :return: (working_dir, handler) tuple, as nuclio expects to get it

    Example: ("a/b/c#main:Handler") -> ("a/b/c", "main:Handler")
    """

    def extend_handler(base_handler):
        # return default handler and module if not specified
        if not base_handler:
            return "main:handler"
        if ":" not in base_handler:
            base_handler = f"{base_handler}:handler"
        return base_handler

    if not handler:
        return "", "main:handler"

    split_handler = handler.split("#")
    if len(split_handler) == 1:
        return "", extend_handler(handler)

    return split_handler[0], extend_handler(split_handler[1])


def _is_nuclio_version_in_range(min_version: str, max_version: str) -> bool:
    """
    Return whether the Nuclio version is in the range, inclusive for min, exclusive for max - [min, max)
    """
    resolved_nuclio_version = None
    try:
        parsed_min_version = semver.VersionInfo.parse(min_version)
        parsed_max_version = semver.VersionInfo.parse(max_version)
        resolved_nuclio_version = mlrun.runtimes.utils.resolve_nuclio_version()
        parsed_current_version = semver.VersionInfo.parse(resolved_nuclio_version)
    except ValueError:
        logger.warning(
            "Unable to parse nuclio version, assuming in range",
            nuclio_version=resolved_nuclio_version,
            min_version=min_version,
            max_version=max_version,
        )
        return True
    return parsed_min_version <= parsed_current_version < parsed_max_version
