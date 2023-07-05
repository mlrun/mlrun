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
import urllib.parse

import semver

import mlrun
import mlrun.api.utils.clients.nuclio
import mlrun.api.utils.runtimes.nuclio
import mlrun.api.utils.singletons.k8s
import mlrun.runtimes
from mlrun.utils import logger


def resolve_function_http_trigger(function_spec):
    for trigger_name, trigger_config in function_spec.get("triggers", {}).items():
        if trigger_config.get("kind") != "http":
            continue
        return trigger_config


def resolve_nuclio_runtime_python_image(
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


def resolve_function_ingresses(function_spec):
    http_trigger = resolve_function_http_trigger(function_spec)
    if not http_trigger:
        return []

    ingresses = []
    for _, ingress_config in (
        http_trigger.get("attributes", {}).get("ingresses", {}).items()
    ):
        ingresses.append(ingress_config)
    return ingresses


def enrich_function_with_ingress(config, mode, service_type):
    # do not enrich with an ingress
    if mode == mlrun.runtimes.constants.NuclioIngressAddTemplatedIngressModes.never:
        return

    ingresses = resolve_function_ingresses(config["spec"])

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


def resolve_function_image_pull_secret(function):
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


def resolve_work_dir_and_handler(handler):
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


def is_nuclio_version_in_range(min_version: str, max_version: str) -> bool:
    """
    Return whether the Nuclio version is in the range, inclusive for min, exclusive for max - [min, max)
    """
    resolved_nuclio_version = None
    try:
        parsed_min_version = semver.VersionInfo.parse(min_version)
        parsed_max_version = semver.VersionInfo.parse(max_version)
        resolved_nuclio_version = (
            mlrun.api.utils.runtimes.nuclio.resolve_nuclio_version()
        )
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


def compile_nuclio_archive_config(
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
    work_dir, handler = resolve_work_dir_and_handler(function.spec.function_handler)
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
