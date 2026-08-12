# Copyright 2026 Iguazio
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
import dataclasses
import os.path
import pathlib
import re
import textwrap
import typing
from collections import defaultdict
from urllib.parse import urlparse

from kubernetes import client

import mlrun.common.constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.model
import mlrun.runtimes
import mlrun.runtimes.mounts
import mlrun.runtimes.pod
import mlrun.runtimes.utils
import mlrun.utils
from mlrun.config import config
from mlrun.k8s_utils import enrich_preemption_mode
from mlrun.utils.helpers import remove_image_protocol_prefix

import framework.utils.helpers
import framework.utils.singletons.k8s

# subdirectory (under an engine's build-context emptyDir) that a fetch-source init container
# writes into, and that a v3io FUSE-mount is nested under - shared so the two backends' Dockerfile
# ``ADD``/``COPY`` source paths agree with what they actually mounted.
FETCHED_SOURCE_SUBDIR = "source"

_DEFAULT_SOURCE_FETCH_IMAGE = "mlrun/mlrun"


@dataclasses.dataclass(frozen=True)
class BuildRequest:
    """The engine-agnostic inputs for a single image build.

    Resolved by the shared build path (:func:`build_image`) and handed to a
    :class:`BuilderBackend`. This is the seam between the shared build flow and
    engine-specific pod construction: the request says *what* to build, and each
    backend decides *how* to stage the source, finalise the Dockerfile's
    ``COPY``/``ADD`` and construct the pod. It is an in-process DTO - not a
    serialized or versioned wire contract.

    ``source`` is carried **raw and unrouted**: routing it to a build context
    (native-remote / http-copy / fetch-init-container / v3io-mount) is engine-owned
    and happens inside :meth:`BuilderBackend.make_build_pod`.

    :param project:            The project the build belongs to.
    :param image_target:       The fully resolved destination image reference.
    :param base_image:         The (enriched) base image the Dockerfile builds ``FROM``.
    :param commands:           Shell commands to run during the build (mlrun-merged).
    :param requirements:       The resolved requirements list to install.
    :param requirements_path:  Path of the requirements file inside the build context.
    :param source:             The raw, unrouted source descriptor / URI.
    :param inline_code:        Inline function code to embed, if any.
    :param inline_path:        Destination filename for ``inline_code``.
    :param extra:              Extra directives appended to the Dockerfile.
    :param builder_env:        The resolved builder-env mapping (used e.g. to derive
                               v3io credentials for a mounted source).
    :param builder_env_list:   ``builder_env`` as build-time env vars for the pod.
    :param project_secrets:    Project secrets exposed as build-time env vars.
    :param extra_args:         Extra builder CLI arguments (validated, engine-rendered
                               inside the backend).
    :param secret_name:        The docker-config secret to authenticate the push.
    :param namespace:          The k8s namespace ``secret_name`` lives in.
    :param registry:           The target registry, when given explicitly.
    :param runtime_spec:       The function's runtime spec (resources, scheduling,
                               security context, ``build`` config).
    :param project_default_function_node_selector: Project-level default node selector.
    :param user_unix_id:       Resolved build UID for the image ``USER`` / ownership.
    :param enriched_group_id:  Resolved build GID for the image ``USER`` / ownership.
    :param auth_info:          The caller's auth info.
    :param name:               The build pod's base name.
    :param labels:             mlrun-internal labels to stamp on the build pod.
    :param verbose:            Whether to run the build engine verbosely.
    """

    project: str
    image_target: str
    base_image: str
    commands: list[str]
    requirements: list[str]
    requirements_path: str
    source: str
    inline_code: str | None
    inline_path: str | None
    extra: str | None
    builder_env: dict[str, str]
    builder_env_list: list[client.V1EnvVar]
    project_secrets: list[client.V1EnvVar]
    extra_args: str | dict
    secret_name: str | None
    namespace: str | None
    registry: str | None
    runtime_spec: typing.Any
    project_default_function_node_selector: dict[str, str]
    user_unix_id: int | None
    enriched_group_id: int | None
    auth_info: mlrun.common.schemas.AuthInfo
    name: str
    labels: dict[str, str]
    verbose: bool


class BuilderBackend(typing.Protocol):
    """A pluggable container-image build engine.

    Each backend turns a :class:`BuildRequest` into a ready-to-run build pod,
    owning everything engine-specific: source routing, the Dockerfile's source
    ``COPY``, engine CLI args, registry auth and the pod's security context. The
    shared build path composes as *resolve the backend for a request, then*
    ``make_build_pod(request)``.
    """

    def make_build_pod(
        self, request: BuildRequest
    ) -> "framework.utils.singletons.k8s.BasePod":
        """Build the (engine-specific) build pod for ``request``.

        :param request: The resolved, engine-agnostic build inputs.
        :return: The build pod, ready for the shared path to launch.
        """
        ...


def resolve_mlrun_install_command_version(
    mlrun_version_specifier=None, client_version=None, commands=None
):
    commands = commands or []
    install_mlrun_regex = re.compile(r".*pip install .*mlrun.*")
    for command in commands:
        if install_mlrun_regex.match(command):
            return None

    unstable_versions = ["unstable", "0.0.0+unstable"]
    unstable_mlrun_version_specifier = (
        f"{config.package_path}[complete] @ git+"
        f"https://github.com/mlrun/mlrun@development"
    )
    if not mlrun_version_specifier:
        if config.httpdb.builder.mlrun_version_specifier:
            mlrun_version_specifier = config.httpdb.builder.mlrun_version_specifier
        elif client_version:
            if client_version not in unstable_versions:
                mlrun_version_specifier = (
                    f"{config.package_path}[complete]=={client_version}"
                )
            else:
                mlrun_version_specifier = unstable_mlrun_version_specifier
        elif config.version in unstable_versions:
            mlrun_version_specifier = unstable_mlrun_version_specifier
        else:
            mlrun_version_specifier = (
                f"{config.package_path}[complete]=={config.version}"
            )
    return mlrun_version_specifier


def resolve_upgrade_pip_command(commands=None):
    commands = commands or []
    pip_upgrade_regex = re.compile(r".*pip install --upgrade .*pip.*")
    for command in commands:
        if pip_upgrade_regex.match(command):
            return None

    return f"python -m pip install --upgrade pip{config.httpdb.builder.pip_version}"


def add_mlrun_to_requirements(build, enriched_base_image, mlrun_version_specifier=None):
    # Add mlrun to the requirements even though it is already installed because
    # we want pip to include mlrun constraints when installing other packages
    image_tag, has_py_package = framework.utils.helpers.extract_image_tag(
        enriched_base_image
    )
    if has_py_package or mlrun_version_specifier:
        installed_mlrun_version_command = resolve_mlrun_install_command_version(
            mlrun_version_specifier, client_version=image_tag
        )
        mlrun.utils.logger.debug(
            "Enriching build requirements with mlrun package",
            enriched_base_image=enriched_base_image,
            installed_mlrun_version_command=installed_mlrun_version_command,
            image_tag=image_tag,
            mlrun_version_specifier=mlrun_version_specifier,
        )
        build.requirements.insert(0, installed_mlrun_version_command)

    else:
        mlrun.utils.logger.warning(
            "Cannot resolve mlrun pypi version from base image, mlrun requirements may be overriden",
            base_image=enriched_base_image,
        )


def is_mlrun_image(base_image):
    mlrun_images = [
        "mlrun/mlrun",
        "mlrun/mlrun-gpu",
        "mlrun/mlrun-kfp",
    ]
    return any([image in base_image for image in mlrun_images])


def resolve_and_enrich_image_target(
    image_target: str,
    registry: str | None = None,
    client_version: str | None = None,
    client_python_version: str | None = None,
) -> str:
    image_target = resolve_image_target(image_target, registry)
    image_target = mlrun.utils.enrich_image_url(
        image_target, client_version, client_python_version
    )
    return image_target


def resolve_image_target(image_target: str, registry: str | None = None) -> str:
    if registry:
        return "/".join([registry, image_target])

    # if dest starts with a dot, we add the configured registry to the start of the dest
    if image_target.startswith(
        mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX
    ):
        # remove prefix from image name
        image_target = image_target[
            len(mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX) :
        ]

        registry, repository = mlrun.utils.get_parsed_docker_registry()
        if not registry:
            raise ValueError(
                "Default docker registry is not defined, set "
                "MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY/MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY_SECRET env vars"
            )
        image_target_components = [registry, image_target]
        if repository and repository not in image_target:
            image_target_components = [registry, repository, image_target]

        return "/".join(image_target_components)

    image_target = remove_image_protocol_prefix(image_target)
    return image_target


def make_dockerfile(
    base_image: str,
    commands: list | None = None,
    source: str | None = None,
    requirements_path: str | None = None,
    target_dir: str = "/mlrun",
    extra: str = "",
    user_unix_id: int | None = None,
    enriched_group_id: int | None = None,
    builder_env: list[client.V1EnvVar] | None = None,
    extra_args: str = "",
    project_secrets: list[client.V1EnvVar] | None = None,
):
    """
    Generates the content of a Dockerfile for building a container image.

    There is a single renderer while Kaniko is the only backend; when a second
    backend needs the source staging done differently, factor out the part that
    actually diverges then (rather than pre-splitting it now).

    :param base_image: The base image for the Dockerfile.
    :param commands: A list of shell commands to be included in the Dockerfile as RUN instructions.
    :param source: The path to the source code directory to be included in the Docker image.
    :param requirements_path: The path to the requirements file (e.g., requirements.txt) containing
                              the Python dependencies to be installed in the Docker image.
    :param target_dir: The directory to which source code will be copied. Default is "/mlrun".
    :param extra: Additional content to be appended to the generated Dockerfile.
    :param user_unix_id: The Unix user ID to be used in the Docker image for running processes.
                         This is useful for matching the user ID with the host environment
                         to avoid permission issues.
    :param enriched_group_id: The group ID to be used in the Docker image for running processes.
    :param builder_env: A list of Kubernetes V1EnvVar objects representing build-time arguments
                        to be set during the build process.
    :param extra_args:  A string containing additional builder arguments in the format of command-line options,
            e.g. extra_args="--skip-tls-verify --build-arg A=val"
    :param project_secrets: A list of Kubernetes V1EnvVar objects representing the project secrets,
            which will be used as build-time arguments in the Dockerfile.
    :return: The content of the Dockerfile as a string.
    """
    dock = f"FROM {base_image}\n"

    builder_env = builder_env or []
    project_secrets = project_secrets or []
    extra_args = _parse_extra_args_for_dockerfile(extra_args)

    # combine a list of all args (including builder_env, project_secrets and extra_args)
    # to add in each of the Dockerfile stages.
    all_args = []
    # Include all builder_env and extra_args as 'ARG arg_name',
    # where the value will be set by the user using the --build-arg flag.
    all_args.extend([env.name for env in builder_env])
    all_args.extend([arg for arg in extra_args])

    # Include project secrets as ARGs, formatted like 'ARG SECRET_NAME=$ARG_NAME',
    # to prevent direct inclusion of the secret as plain text within the Dockerfile.
    all_args.extend([f"{secret.name}=${secret.name}" for secret in project_secrets])

    # add all args to the dockerfile
    args = ""
    for arg in all_args:
        args += f"ARG {arg}\n"
    dock += args

    if config.is_pip_ca_configured():
        dock += f"COPY ./{pathlib.Path(config.httpdb.builder.pip_ca_path).name} {config.httpdb.builder.pip_ca_path}\n"
        dock += f"ARG PIP_CERT={config.httpdb.builder.pip_ca_path}\n"

    build_args = config.get_build_args()
    for build_arg_key, build_arg_value in build_args.items():
        dock += f"ARG {build_arg_key}={build_arg_value}\n"

    if source:
        args = args.rstrip("\n")
        # 'ADD' command does not extract zip files - add extraction stage to the dockerfile
        # it is up to base image to have unzip included in case source is zip
        if source.endswith(".zip"):
            source_dir = os.path.join(target_dir, "source")
            filename = os.path.basename(source)
            stage_lines = [
                f"FROM {base_image} AS extractor",
                args,
                f"RUN mkdir -p {source_dir}",
                f"ADD {source} {source_dir}",
                f"RUN cd {source_dir} && unzip {filename} && rm {filename}",
            ]
            stage = textwrap.dedent("\n".join(stage_lines)).strip()
            dock = stage + "\n" + dock

            dock += f"COPY --from=extractor {source_dir}/ {target_dir}\n"
        else:
            dock += f"ADD {source} {target_dir}\n"

        if user_unix_id is not None and enriched_group_id is not None:
            dock += f"RUN chown -R {user_unix_id}:{enriched_group_id} {target_dir}\n"

        dock += f"ENV PYTHONPATH {target_dir}\n"
    if commands:
        dock += "".join([f"RUN {command}\n" for command in commands])
    if requirements_path:
        dock += (
            f"RUN echo 'Installing {requirements_path}...'; cat {requirements_path}\n"
        )
        dock += f"RUN python -m pip install -r {requirements_path}\n"
    if extra:
        dock += extra
    mlrun.utils.logger.debug("Resolved dockerfile", dockfile_contents=dock)

    return dock


def build_pod_resources(runtime_spec) -> dict:
    """Resolve the build-pod resource policy shared by every builder backend.

    Requests-only + GPU-limit-zero, in one place so it can't drift between engines:

    * Only **requests** are set. Requests affect scheduling; setting a limit could kill the
      build mid-run (destructive), so the build pod is never given a limit for cpu/memory.
    * A **zero GPU limit** is added when the function requests GPUs. Some cloud providers add a
      toleration only when a GPU limit is present; without it a build pod that inherited a
      GPU-related node selector from the function could stay pending. Zero keeps the toleration
      applied while allocating no GPU.

    :param runtime_spec: The function's runtime spec (for its resource limits), or ``None``.
    :return: A resources dict with ``requests`` (and ``limits`` only when GPUs are requested).
    """
    # we cannot specify gpu requests without specifying gpu limits, so we set requests without gpu field
    default_requests = config.get_default_function_pod_requirement_resources(
        "requests", with_gpu=False
    )
    resources = {
        "requests": mlrun.runtimes.utils.generate_resources(
            mem=default_requests.get("memory"), cpu=default_requests.get("cpu")
        )
    }
    if runtime_spec:
        gpu_resources = mlrun.utils.get_enriched_gpu_limits(
            runtime_spec.resources.get("limits", {})
        )
        if gpu_resources:
            resources["limits"] = gpu_resources
    return resources


def resolve_builder_pod_labels(extra_labels: dict | None) -> dict:
    """Merge the configured builder-pod labels under the caller's labels.

    Shared by every builder backend so the precedence can't drift between engines. The
    configured ``pod_labels`` (e.g. ``azure.workload.identity/use``, which lets the Azure
    workload-identity webhook inject ACR push credentials) are the lowest-precedence layer:
    MLRun's own internal labels (``mlrun/class``, ``mlrun/project``, ...) must never be clobbered
    by them, so they are filtered out and the caller's ``extra_labels`` win on any conflict.

    :param extra_labels: The backend/caller labels, which take precedence.
    :return: The merged label dict for the build pod.
    """
    configured_pod_labels = {
        key: value
        for key, value in config.get_builder_pod_labels().items()
        if key not in mlrun.common.constants.MLRunInternalLabels.all()
    }
    return mlrun.utils.helpers.merge_dicts_with_precedence(
        configured_pod_labels,
        extra_labels or {},
    )


def resolve_build_pod_spec_attributes(
    project: str,
    runtime_spec,
    project_default_function_node_selector,
    auth_info: mlrun.common.schemas.AuthInfo | None = None,
) -> dict:
    """Resolve the runtime-derived pod-spec attributes shared by every builder backend.

    Reads the function's runtime spec and returns the pod-spec attributes the build pod should
    carry - node name/selector, affinity, tolerations, priority class and service account - applying
    the same preemption-mode enrichment and service-account resolution a regular function pod gets.
    Engine-agnostic: both Kaniko and Buildah apply the result via
    ``BasePod.default_pod_spec_attributes``, so scheduling/identity can't drift between engines.

    :param project:            The project the build belongs to.
    :param runtime_spec:       The function's runtime spec.
    :param project_default_function_node_selector: Project-level default node selector.
    :param auth_info:          The caller's auth info (for service-account resolution).
    :return: The resolved pod-spec attributes (only non-empty values).
    """
    # preemption mode scheduling constraints cache
    _preemption_enrichment_result = {}

    def service_account_handler(attr_value):
        from framework.api.utils import resolve_project_service_account_details

        (
            allowed_service_accounts,
            forbidden_service_accounts,
            default_service_account,
        ) = resolve_project_service_account_details(project, auth_info=auth_info)
        if attr_value:
            runtime_spec.validate_service_account(
                allowed_service_accounts, forbidden_service_accounts
            )
        else:
            attr_value = default_service_account
        return attr_value

    def get_merged_node_selector(attr_value):
        return mlrun.utils.to_non_empty_values_dict(
            mlrun.utils.helpers.merge_dicts_with_precedence(
                mlrun.mlconf.get_default_function_node_selector(),
                project_default_function_node_selector,
                attr_value,
            )
        )

    def preemption_mode_handler(key):
        if key not in _preemption_enrichment_result:
            keys = ["node_selector", "tolerations", "affinity"]
            values = enrich_preemption_mode(
                preemption_mode=runtime_spec.preemption_mode,
                node_selector=get_merged_node_selector(runtime_spec.node_selector),
                affinity=runtime_spec.affinity,
                tolerations=runtime_spec.tolerations,
            )
            _preemption_enrichment_result.update(dict(zip(keys, values)))
        return _preemption_enrichment_result[key]

    def node_selector_handler(attr_value):
        return preemption_mode_handler("node_selector")

    def affinity_handler(attr_value):
        return preemption_mode_handler("affinity")

    def tolerations_handler(attr_value):
        return preemption_mode_handler("tolerations")

    def identity_handler(attr_value):
        return attr_value

    # spec attributes defined on the runtime that should also apply to the build pod, each with the
    # handler that resolves its value.
    handlers = {
        "node_name": identity_handler,
        "node_selector": node_selector_handler,
        "affinity": affinity_handler,
        "tolerations": tolerations_handler,
        "priority_class_name": identity_handler,
        "service_account": service_account_handler,
    }

    resolved = {}
    for attribute, handler in handlers.items():
        attr_value = handler(getattr(runtime_spec, attribute, None))
        if attr_value:
            resolved[attribute] = attr_value
    return resolved


def normalize_git_source_fragment(source: str) -> str:
    """Normalize a ``git://`` source's fragment to a full ``refs/...`` ref.

    A user-supplied branch/tag fragment without the ``refs/heads/``/``refs/tags/`` prefix is
    assumed to be a branch. Shared by every backend that resolves a ``git://`` source itself
    (rather than delegating the whole URI to the engine), so the assumption can't drift.

    :param source: The raw ``git://...#<fragment>`` source descriptor.
    :return: ``source`` with its fragment normalized, unchanged if there is no fragment or it
        already starts with ``refs/``.
    """
    fragment = urlparse(source).fragment or ""
    if fragment and not fragment.startswith("refs/"):
        source = source.replace("#" + fragment, f"#refs/heads/{fragment}")
    return source


def resolve_source_code_target_dir(
    request: BuildRequest, source_to_copy: str | None
) -> None:
    """Resolve the in-image source target dir (mutates the runtime build spec).

    Only relevant when there is source to copy; a relative or unset target dir is anchored under
    ``/home/mlrun_code``. Shared by every backend so the anchoring can't drift between engines.

    :param request:        The build request (its ``runtime_spec`` is mutated).
    :param source_to_copy: The routed source, or ``None`` when nothing is copied.
    """
    source_code_target_dir = request.runtime_spec.build.source_code_target_dir
    if source_to_copy and (
        not source_code_target_dir or not os.path.isabs(source_code_target_dir)
    ):
        relative_workdir = source_code_target_dir or ""
        relative_workdir = relative_workdir.removeprefix("./")

        request.runtime_spec.build.source_code_target_dir = os.path.join(
            "/home/mlrun_code", relative_workdir
        )


def mount_v3io_source(
    request: BuildRequest,
    pod: "framework.utils.singletons.k8s.BasePod",
    v3io_dir_to_mount: str,
    mount_path: str,
) -> None:
    """Mount a v3io source directory into a build pod's build context.

    Shared by every backend that supports a v3io source, so the credential-resolution precedence
    (``builder_env`` overrides, falling back to ``auth_info``) can't drift between engines - only
    the mount path (each engine's own build-context layout) is engine-specific.

    :param request:           The build request (for v3io credentials).
    :param pod:               The build pod to mount into.
    :param v3io_dir_to_mount: The normalized v3io directory to mount.
    :param mount_path:        Where to mount it inside the pod (engine-specific).
    """
    access_key = request.builder_env.get(
        "V3IO_ACCESS_KEY",
        request.auth_info.data_session or request.auth_info.access_key,
    )
    username = request.builder_env.get("V3IO_USERNAME", request.auth_info.username)
    pod.mount_v3io(
        remote=v3io_dir_to_mount,
        mount_path=mount_path,
        access_key=access_key,
        user=username,
    )


def append_source_fetch_init_container(
    pod: "framework.utils.singletons.k8s.BasePod",
    source: str,
    target_dir: str,
    builder_env_list: list,
    project_secrets: list,
) -> None:
    """Append a ``fetch-source`` init container that runs ``mlrun load-source``.

    Fetches ``source`` into ``target_dir`` on a volume already mounted into ``pod`` (every init
    container inherits the pod's volume mounts). Shared by every backend that stages a remote
    source locally before its build step runs.

    :param pod:              The build pod to append the init container to.
    :param source:           The raw source URI to fetch.
    :param target_dir:       Where ``mlrun load-source`` should write the fetched source, inside
                             a volume the build container also has mounted.
    :param builder_env_list: Build-time env vars (highest precedence).
    :param project_secrets:  Project secrets exposed as build-time env vars (next precedence).
    """
    # Env precedence: builder_env_list > project_secrets > storage.auto_mount_params.
    # First-write wins so caller-supplied values are not overwritten by auto-mount defaults.
    image = config.httpdb.builder.kaniko_source_fetch_init_container_image
    if not image:
        image = mlrun.utils.enrich_image_url(_DEFAULT_SOURCE_FETCH_IMAGE)

    args = ["-m", "mlrun", "load-source", source, "--target", target_dir]

    env_list = list(builder_env_list or []) + list(project_secrets or [])
    already_set = {env_var.name for env_var in env_list}
    for env_var in resolve_storage_auto_mount_env():
        if env_var.name in already_set:
            continue
        env_list.append(env_var)
        already_set.add(env_var.name)

    # only the scheme is logged - a source URI can embed credentials
    # (e.g. https://<token>@github.com/... or a presigned s3/http URL), which must never be logged.
    mlrun.utils.logger.debug(
        "Adding source-fetch init container",
        image=image,
        source_scheme=urlparse(source).scheme or "local",
        target=target_dir,
    )
    pod.append_init_container(
        image,
        command=["python"],
        args=args,
        env=env_list,
        name="fetch-source",
    )


def resolve_storage_auto_mount_env() -> list:
    # Gate on env_style_modifiers so mount-style outputs (volumes/volume_mounts) are
    # not silently dropped. KubeResource.apply sanitizes spec.env to plain dicts, so
    # rebuild V1EnvVar for callers that rely on attribute access.
    auto_mount_type = mlrun.runtimes.pod.AutoMountType(
        mlrun.mlconf.storage.auto_mount_type
    )
    modifier = auto_mount_type.get_modifier()
    if (
        modifier is None
        or modifier not in mlrun.runtimes.pod.AutoMountType.env_style_modifiers()
    ):
        return []
    scratch = mlrun.runtimes.KubejobRuntime()
    scratch.try_auto_mount_based_on_config()
    return [
        client.V1EnvVar(
            name=env_var["name"],
            value=env_var.get("value"),
            value_from=env_var.get("valueFrom"),
        )
        if isinstance(env_var, dict)
        else env_var
        for env_var in scratch.spec.env or []
    ]


def _generate_builder_env(
    project: str, builder_env: dict
) -> (list[client.V1EnvVar], list[client.V1EnvVar]):
    k8s = framework.utils.singletons.k8s.get_k8s_helper(silent=False)
    secret_name = k8s.get_project_secret_name(project)
    existing_secret_keys = k8s.get_project_secret_keys(project, filter_internal=True)

    # generate env list from builder env and project secrets
    project_secrets = []
    for key in existing_secret_keys:
        if key not in builder_env:
            value_from = client.V1EnvVarSource(
                secret_key_ref=client.V1SecretKeySelector(name=secret_name, key=key)
            )
            project_secrets.append(client.V1EnvVar(name=key, value_from=value_from))
    env = []
    for key, value in builder_env.items():
        env.append(client.V1EnvVar(name=key, value=value))
    return env, project_secrets


def _parse_extra_args_for_dockerfile(extra_args: str) -> dict:
    if not extra_args:
        return {}

    build_arg_values = {}
    is_build_arg = False

    for arg in extra_args.split():
        if arg == "--build-arg":
            is_build_arg = True
        elif arg.startswith("--"):
            is_build_arg = False
        elif is_build_arg:
            # Ensure 'arg' is in a valid format: starts with a letter or underscore,
            # followed by alphanumerics and an equal sign
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=[^=]+$", arg):
                raise ValueError(f"Invalid --build-arg value: {arg}")
            key, val = arg.split("=")
            build_arg_values[key] = val
        else:
            is_build_arg = False

    return build_arg_values


def _resolve_build_requirements(
    requirements: typing.Union[list, str],
    commands: list,
    with_mlrun: bool,
    mlrun_version_specifier: str | None,
    client_version: str | None,
):
    """
    Resolve build requirements list, requirements path and commands.
    If mlrun requirement is needed, we add a pip upgrade command to the commands list (prerequisite).
    """
    requirements_path = "/empty/requirements.txt"
    if requirements and isinstance(requirements, list):
        requirements_list = requirements
    else:
        requirements_list = []
        requirements_path = requirements or requirements_path
    commands = commands or []

    if with_mlrun:
        # mlrun prerequisite - upgrade pip
        upgrade_pip_command = resolve_upgrade_pip_command(commands)
        if upgrade_pip_command:
            commands.append(upgrade_pip_command)

        mlrun_version = resolve_mlrun_install_command_version(
            mlrun_version_specifier, client_version, commands
        )

        # mlrun must be installed with other python requirements in the same pip command to avoid version conflicts
        if mlrun_version:
            requirements_list.insert(0, mlrun_version)

    if not requirements_list:
        # no requirements, we don't need a requirements file
        requirements_path = ""

    return commands, requirements_list, requirements_path


def _parse_extra_args(extra_args: str) -> dict:
    """
    Parses a string of extra arguments into a dictionary format.

    :param extra_args:  A string containing additional builder arguments in the format of command-line options,
            e.g. extra_args="--skip-tls-verify --build-arg A=val"

    :return: A dictionary where each key corresponds to an option flag (e.g., "--option_name"),
             and the associated value is a list of values provided for that option.

    :example:
    >>> extra_args = "--option1 value1 --option2 value3 --option3 --option1 value2"
    >>> parsed_args = _parse_extra_args(extra_args)
    >>> print(parsed_args)
    {
        '--option1': ['value1', 'value2'],
        '--option2': ['value3'],
        '--option3': []
    }
    """
    if not extra_args:
        return {}
    extra_args = extra_args.split()
    args = defaultdict(list)

    current_flag = None
    for arg in extra_args:
        if arg.startswith("--"):
            current_flag = arg
            # explicitly set the key in the dictionary
            args.setdefault(current_flag, [])
        elif current_flag:
            args[current_flag].append(arg)

        # sanity, args should be validated by now
        else:
            raise ValueError(
                "Invalid argument sequence. Value must be followed by a flag preceding it."
            )
    return args


def _validate_extra_args(extra_args: str):
    """
     Validate extra_args string for Docker commands:
    - Ensure --build-arg is followed by a non-flag argument.
    - Validate all --build-arg values are in a valid format of 'KEY=VALUE' using allowed characters only.

    :raises ValueError: If the extra_args sequence is invalid or contains incorrectly formatted '--build-arg' values.
    """
    if not extra_args:
        return

    if not extra_args.startswith("--"):
        raise ValueError(
            "Invalid argument sequence. Value must be followed by a flag preceding it."
        )
    args = _parse_extra_args(extra_args)
    for arg, values in args.items():
        if arg == "--build-arg":
            if not values:
                raise ValueError(
                    "Invalid '--build-arg' usage. It must be followed by a non-flag argument."
                )
            invalid_build_arg_values = [
                val
                for val in values
                if not re.match(r"^[a-zA-Z0-9_]+=[a-zA-Z0-9_]+$", val)
            ]
            if invalid_build_arg_values:
                raise ValueError(
                    f"Invalid arguments format: '{','.join(invalid_build_arg_values)}'."
                    " Please make sure all arguments are in a valid format"
                )


def _resolve_function_image_name(function, image: str | None = None) -> str:
    project = function.metadata.project
    name = function.metadata.name
    tag = function.metadata.tag or "latest"
    if image:
        image_name_prefix = (
            mlrun.runtimes.utils.resolve_function_target_image_name_prefix(
                project, name
            )
        )
        registries_to_enforce_prefix = mlrun.runtimes.utils.resolve_function_target_image_registries_to_enforce_prefix()
        for registry in registries_to_enforce_prefix:
            if image.startswith(registry):
                prefix_with_registry = f"{registry}{image_name_prefix}"
                if not image.startswith(prefix_with_registry):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Configured registry enforces image name to start with this prefix: {image_name_prefix}"
                    )
        return image
    return _generate_function_image_name(project, name, tag)


def _generate_function_image_name(project: str, name: str, tag: str) -> str:
    _, repository = mlrun.utils.get_parsed_docker_registry()
    repository = mlrun.utils.helpers.get_docker_repository_or_default(repository)
    return mlrun.runtimes.utils.fill_function_image_name_template(
        mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX,
        repository,
        project,
        name,
        tag,
    )


def _resolve_function_image_secret(
    resolved_target_image: str, secret: str | None = None
) -> str:
    if not secret:
        parsed_registry, _ = mlrun.utils.get_parsed_docker_registry()

        # populate default secret if target image prefix equals to either the implicit or explicit default registry
        if (
            parsed_registry and resolved_target_image.startswith(parsed_registry)
        ) or resolved_target_image.startswith(
            mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX
        ):
            secret = config.httpdb.builder.docker_registry_secret
    return secret
