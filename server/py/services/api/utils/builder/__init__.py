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
from base64 import b64decode

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.model
import mlrun.runtimes
import mlrun.utils
from mlrun.config import config

import framework.utils.singletons.k8s
from services.api.utils.builder.base import (
    BuilderBackend,
    BuildRequest,
    _generate_builder_env,
    _resolve_build_requirements,
    _resolve_function_image_name,
    _resolve_function_image_secret,
    _validate_extra_args,
    add_mlrun_to_requirements,
    is_mlrun_image,
    resolve_and_enrich_image_target,
    resolve_image_target,
    resolve_mlrun_install_command_version,
)
from services.api.utils.builder.buildah import BuildahBackend
from services.api.utils.builder.kaniko import KanikoBackend


def build_image(
    auth_info: mlrun.common.schemas.AuthInfo,
    project: str,
    image_target,
    commands=None,
    source="",
    base_image=None,
    requirements=None,
    inline_code=None,
    inline_path=None,
    secret_name=None,
    namespace=None,
    with_mlrun=True,
    mlrun_version_specifier=None,
    registry=None,
    interactive=True,
    name="",
    extra=None,
    verbose=False,
    builder_env=None,
    client_version=None,
    runtime=None,
    extra_args=None,
    force_build=None,
):
    runtime_spec = runtime.spec if runtime else None
    runtime_builder_env = runtime_spec.build.builder_env or {}

    project_default_function_node_selector = {}
    if runtime and runtime._get_db():
        project_obj = runtime._get_db().get_project(runtime.metadata.project)
        if project_obj:
            project_default_function_node_selector = (
                project_obj.spec.default_function_node_selector
            )

    extra_args = extra_args or {}
    builder_env = builder_env or {}

    builder_env = runtime_builder_env | builder_env or {}
    # no need to enrich extra args because we get them from the build anyway
    _validate_extra_args(extra_args)

    image_target = resolve_image_target(image_target, registry)
    commands, requirements_list, requirements_path = _resolve_build_requirements(
        requirements, commands, with_mlrun, mlrun_version_specifier, client_version
    )

    if force_build:
        mlrun.utils.logger.info("Forcefully building image")
    elif not inline_code and not source and not commands and not requirements:
        mlrun.utils.logger.info("Skipping build, nothing to add")
        return "skipped"

    builder_env_list, project_secrets = _generate_builder_env(project, builder_env)

    user_unix_id = None
    enriched_group_id = None
    if (
        mlrun.mlconf.function.spec.security_context.enrichment_mode
        != mlrun.common.schemas.SecurityContextEnrichmentModes.disabled.value
    ):
        from framework.api.utils import ensure_function_security_context

        ensure_function_security_context(runtime, auth_info)
        user_unix_id = runtime.spec.security_context.run_as_user
        enriched_group_id = runtime.spec.security_context.run_as_group

    # everything above is engine-agnostic resolution. Package it into the seam DTO,
    # then let the resolved backend own source routing, the Dockerfile source COPY
    # and pod construction (see BuilderBackend / BuildRequest).
    request = BuildRequest(
        project=project,
        image_target=image_target,
        base_image=base_image,
        commands=commands,
        requirements=requirements_list,
        requirements_path=requirements_path,
        source=source,
        inline_code=inline_code,
        inline_path=inline_path,
        extra=extra,
        builder_env=builder_env,
        builder_env_list=builder_env_list,
        project_secrets=project_secrets,
        extra_args=extra_args,
        secret_name=secret_name,
        registry=registry,
        runtime_spec=runtime_spec,
        project_default_function_node_selector=project_default_function_node_selector,
        user_unix_id=user_unix_id,
        enriched_group_id=enriched_group_id,
        auth_info=auth_info,
        name=name,
        labels={
            mlrun_constants.MLRunInternalLabels.name: name,
            mlrun_constants.MLRunInternalLabels.function: runtime.metadata.name,
            mlrun_constants.MLRunInternalLabels.tag: runtime.metadata.tag or "latest",
        },
        verbose=verbose,
    )

    backend = resolve_builder_backend(request)
    build_pod = backend.make_build_pod(request)

    k8s = framework.utils.singletons.k8s.get_k8s_helper(silent=False)
    build_pod.namespace = k8s.resolve_namespace(namespace)

    if interactive:
        return k8s.run_job(build_pod)
    else:
        pod, ns = k8s.create_pod(build_pod)
        mlrun.utils.logger.info(
            "Build started", pod=pod, namespace=ns, project=project, image=image_target
        )
        return f"build:{pod}"


def resolve_builder_backend(request: BuildRequest) -> BuilderBackend:
    """Return the builder backend to use for a build request.

    By default this is the engine named in ``httpdb.builder.builder_backend``. The whole
    ``request`` is accepted so the choice can vary per build: when Buildah is configured but the
    request needs a capability the Buildah adapter doesn't ship yet, this transparently falls back
    to Kaniko (see :func:`_buildah_fallback_reason`).

    :param request: The resolved build request.
    :return: The builder backend instance for this build.
    :raises mlrun.errors.MLRunInvalidArgumentError: If the configured backend is unknown.
    """
    # keyed by the httpdb.builder.builder_backend config value; future engines register here
    # without touching the shared path.
    backends: dict[str, type[BuilderBackend]] = {
        "kaniko": KanikoBackend,
        "buildah": BuildahBackend,
    }
    backend_name = config.httpdb.builder.builder_backend
    backend_class = backends.get(backend_name)
    if backend_class is None:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Unsupported builder backend '{backend_name}'. "
            f"Supported backends: {', '.join(sorted(backends))}"
        )

    # Buildah is opt-in and, as of ML-12885, ships the rootless build pod but not yet the full
    # runnable path. Fall back to Kaniko for the inputs it can't handle yet so a buildah-configured
    # cluster never emits a pod that can't build or push.
    if backend_class is BuildahBackend:
        fallback_reason = _buildah_fallback_reason(request)
        if fallback_reason:
            mlrun.utils.logger.info(
                "Builder backend falling back to Kaniko for an unsupported build",
                requested_backend=backend_name,
                reason=fallback_reason,
            )
            return KanikoBackend()

    return backend_class()


def build_runtime(
    auth_info: mlrun.common.schemas.AuthInfo,
    runtime: mlrun.runtimes.BaseRuntime,
    with_mlrun=True,
    mlrun_version_specifier=None,
    skip_deployed=False,
    interactive=False,
    builder_env=None,
    client_version=None,
    client_python_version=None,
    force_build=False,
):
    build: mlrun.model.ImageBuilder = runtime.spec.build
    namespace = runtime.metadata.namespace
    project = runtime.metadata.project
    if skip_deployed and runtime.is_deployed():
        mlrun.utils.logger.info(
            "Skipping build, runtime is already deployed",
            runtime_name=runtime.metadata.name,
            project=project,
        )
        runtime.status.state = mlrun.common.schemas.FunctionState.ready
        return True

    base_image: str = build.base_image or runtime.spec.image
    if not base_image:
        base_image = mlrun.mlconf.function_defaults.image_by_kind.to_dict().get(
            runtime.kind, config.default_base_image
        )

    mlrun_image = False
    # If the base is one of mlrun images - set with_mlrun to False, so it won't be added later
    if base_image and is_mlrun_image(base_image):
        mlrun_image = True
        with_mlrun = False

    if force_build:
        mlrun.utils.logger.info("Forcefully building image")
    elif (
        not build.source
        and not build.commands
        and not build.requirements
        and not build.extra
        and not with_mlrun
    ):
        if not runtime.spec.image:
            if build.base_image:
                runtime.spec.image = build.base_image
            elif runtime.kind in mlrun.mlconf.function_defaults.image_by_kind.to_dict():
                runtime.spec.image = (
                    mlrun.mlconf.function_defaults.image_by_kind.to_dict()[runtime.kind]
                )
        if not runtime.spec.image:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "The deployment was not successful because no image was specified or there are missing build parameters"
                " (commands/source)"
            )

        runtime.status.state = mlrun.common.schemas.FunctionState.ready
        return True

    build.image = _resolve_function_image_name(runtime, build.image)

    # config.httpdb.builder.docker_registry_secret
    build.secret = _resolve_function_image_secret(build.image, build.secret)
    runtime.status.state = ""

    inline = None  # noqa: F841
    if build.functionSourceCode:
        inline = b64decode(build.functionSourceCode).decode("utf-8")  # noqa: F841
    if not build.image:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Build spec must have a target image, set build.image = <target image>"
        )
    name = mlrun.utils.normalize_name(f"mlrun-build-{runtime.metadata.name}")

    enriched_base_image = runtime.full_image_path(
        base_image, client_version, client_python_version
    )

    if mlrun_image and build.requirements:
        add_mlrun_to_requirements(build, enriched_base_image, mlrun_version_specifier)

    mlrun.utils.logger.info(
        "Building runtime image",
        base_image=enriched_base_image,
        image=build.image,
        project=project,
        name=name,
    )

    status = build_image(
        auth_info,
        project,
        image_target=build.image,
        base_image=enriched_base_image,
        commands=build.commands,
        requirements=build.requirements,
        namespace=namespace,
        source=build.source,
        secret_name=build.secret,
        interactive=interactive,
        name=name,
        with_mlrun=with_mlrun,
        mlrun_version_specifier=mlrun_version_specifier,
        extra=build.extra,
        extra_args=build.extra_args,
        verbose=runtime.verbose,
        builder_env=builder_env,
        client_version=client_version,
        runtime=runtime,
        force_build=force_build,
    )
    runtime.status.build_pod = None
    if status == "skipped":
        # using enriched base image for the runtime spec image, because this will be the image that the function will
        # run with
        runtime.spec.image = enriched_base_image
        runtime.status.state = mlrun.common.schemas.FunctionState.ready
        return True

    if status.startswith("build:"):
        runtime.status.state = mlrun.common.schemas.FunctionState.deploying
        runtime.status.build_pod = status[6:]
        # using the base_image, and not the enriched one so we won't have the client version in the image, useful for
        # exports and other cases where we don't want to have the client version in the image, but rather enriched on
        # API level
        runtime.spec.build.base_image = base_image
        return False

    mlrun.utils.logger.info("Build completed", status=status)
    if status in ["failed", "error"]:
        runtime.status.state = mlrun.common.schemas.FunctionState.error
        return False

    local = "" if build.secret or build.image.startswith(".") else "."
    runtime.spec.image = local + build.image
    runtime.status.state = mlrun.common.schemas.FunctionState.ready
    return True


def _buildah_fallback_reason(request: BuildRequest) -> str | None:
    """Return why a Buildah build must fall back to Kaniko, or ``None`` if Buildah can handle it.

    Each guard is temporary — it exists only until its follow-up ships the missing capability, and
    should be removed when that ticket merges.

    :param request: The resolved build request.
    :return: A human-readable fallback reason, or ``None`` when Buildah can build the request.
    """
    # Cloud-registry credential-helper auth -> ML-12886. The Buildah adapter authenticates only
    # via a mounted static docker-config secret; ECR/ACR/GAR workload-identity credential
    # exchange is wired in ML-12886. Remove this guard when ML-12886 merges.
    # (The registry host is a plain hostname - safe to log.)
    target = request.registry or request.image_target or ""
    if _is_cloud_registry(target):
        return (
            f"target registry '{target}' requires credential-helper auth not yet supported "
            "on Buildah (ML-12886)"
        )

    return None


def _is_cloud_registry(target: str) -> bool:
    # cloud registries whose auth needs a credential-helper token exchange (ML-12886).
    if not target:
        return False
    if mlrun.utils.helpers.is_ecr_url(target):
        return True
    # ACR (Azure) and Artifact Registry / GCR (Google).
    return any(
        marker in target for marker in (".azurecr.io", "-docker.pkg.dev", "gcr.io")
    )
