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
import os.path
import pathlib
import shlex
from base64 import b64encode
from urllib.parse import urlparse

from kubernetes import client

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils
import mlrun.utils.clones
from mlrun.config import config

import framework.utils.singletons.k8s
from services.api.utils.builder import base

# the rootless build runs as this uid/gid regardless of the function's security-context enrichment
# (D13): the stock quay.io/buildah/stable image ships /etc/subuid + /etc/subgid ranges for its
# "build" user (uid 1000), which is what the caps rootless model maps from.
_BUILD_UID = 1000
_BUILD_GID = 1000

# the build image's home; buildah keeps its container store under $HOME/.local/share/containers.
_BUILD_HOME = "/home/build"
_CONTAINERS_STORE = f"{_BUILD_HOME}/.local/share/containers"

# where the Dockerfile (and any inline code / requirements) are staged and used as the build context.
_CONTEXT_DIR = "/empty"

# static docker-config secret mount (the non-cloud registry auth path). Cloud-registry credential
# helpers are wired in a follow-up (ML-12886); resolve_builder_backend falls back to Kaniko until then.
_AUTHFILE_DIR = "/auth"
_AUTHFILE_PATH = f"{_AUTHFILE_DIR}/config.json"

# the AppArmor profile is applied via a per-container annotation (the k8s client in the test image is
# capped below the securityContext.appArmorProfile field by KFP v1, and the annotation is also honored
# by pre-1.30 clusters). The annotation is keyed by the build container's name; we read that name from
# the pod itself (BasePod.container_name) so there is a single source of truth and it can't drift.
_APPARMOR_ANNOTATION_PREFIX = "container.apparmor.security.beta.kubernetes.io"

_SUPPORTED_STORAGE_DRIVERS = ("overlay", "vfs")


class BuildahBackend:
    """A rootless `Buildah <https://buildah.io/>`_ build backend behind the
    :class:`~services.api.utils.builder.base.BuilderBackend` seam.

    Builds the same Dockerfile as Kaniko (via :func:`base.make_dockerfile`) in a rootless,
    non-privileged pod on the stock ``quay.io/buildah/stable`` image: ``buildah bud`` then
    ``buildah push``, with ``BUILDAH_ISOLATION=chroot`` and the storage driver from
    ``httpdb.builder.buildah_storage_driver``. The rootless model is fixed to the ``caps`` model
    (``SETUID``/``SETGID`` + ``allowPrivilegeEscalation``); the ``hostUsers`` model was dropped after
    POC-1 showed it is not viable on the target runtimes.

    Scope note: this adapter handles static docker-config-secret registry auth only.
    Cloud-registry credential helpers are wired in a follow-up (ML-12886);
    :func:`~services.api.utils.builder.resolve_builder_backend` transparently falls back to Kaniko
    for those inputs until it lands.

    Source acquisition: Buildah's ``bud --context`` has no native remote-context resolution
    (unlike Kaniko's git/s3 ``--context`` and Dockerfile ``ADD``-from-URL for http), so every
    remote source is either fetched via the ``fetch-source`` init container (git, archives, s3,
    http(s)) or, for v3io, FUSE-mounted - both write/mount into
    ``{_CONTEXT_DIR}/{base.FETCHED_SOURCE_SUBDIR}``, the same emptyDir already mounted for
    Dockerfile/inline-code staging.
    """

    def make_build_pod(
        self, request: base.BuildRequest
    ) -> framework.utils.singletons.k8s.BasePod:
        """Build the rootless Buildah build pod for ``request``.

        :param request: The resolved, engine-agnostic build inputs.
        :return: The Buildah build pod.
        """
        source_to_copy, source_to_fetch, v3io_dir_to_mount = self._route_source(request)
        base.resolve_source_code_target_dir(request, source_to_copy=source_to_copy)

        dockerfile = base.make_dockerfile(
            request.base_image,
            request.commands,
            source=source_to_copy,
            requirements_path=request.requirements_path,
            extra=request.extra,
            target_dir=request.runtime_spec.build.source_code_target_dir,
            builder_env=request.builder_env_list,
            project_secrets=request.project_secrets,
            extra_args=request.extra_args,
        )
        buildah_pod = make_buildah_pod(
            project=request.project,
            dest=request.image_target,
            dockerfile=dockerfile,
            inline_code=request.inline_code,
            inline_path=request.inline_path,
            requirements=request.requirements,
            requirements_path=request.requirements_path,
            secret_name=request.secret_name,
            name=request.name,
            verbose=request.verbose,
            builder_env=request.builder_env_list,
            project_secrets=request.project_secrets,
            runtime_spec=request.runtime_spec,
            registry=request.registry,
            extra_labels=request.labels,
            project_default_function_node_selector=request.project_default_function_node_selector,
            auth_info=request.auth_info,
        )

        if source_to_fetch:
            base.append_source_fetch_init_container(
                pod=buildah_pod,
                source=source_to_fetch,
                target_dir=f"{_CONTEXT_DIR}/{base.FETCHED_SOURCE_SUBDIR}",
                builder_env_list=request.builder_env_list,
                project_secrets=request.project_secrets,
            )
        elif v3io_dir_to_mount:
            base.mount_v3io_source(
                request,
                buildah_pod,
                v3io_dir_to_mount,
                mount_path=f"{_CONTEXT_DIR}/{base.FETCHED_SOURCE_SUBDIR}",
            )

        return buildah_pod

    @staticmethod
    def _route_source(
        request: base.BuildRequest,
    ) -> tuple[str | None, str | None, str | None]:
        """Route the raw source descriptor to the Buildah build context.

        Buildah has no native remote-context resolution, so - unlike Kaniko - every remote source
        is either fetched or FUSE-mounted; none is ever passed through as a raw remote URI.

        :param request: The build request carrying the raw ``source``.
        :return: ``(source_to_copy, source_to_fetch, v3io_dir_to_mount)``. At most one of
            ``source_to_fetch``/``v3io_dir_to_mount`` is set.
        """
        source = request.source
        loads_source_on_run = bool(
            request.runtime_spec and request.runtime_spec.build.load_source_on_run
        )
        if request.inline_code or loads_source_on_run or not source:
            return None, None, None

        if source.startswith("v3io://") or source.startswith("v3ios://"):
            v3io_path = urlparse(source).path
            v3io_dir_to_mount, basename = os.path.split(v3io_path)
            v3io_dir_to_mount = os.path.normpath(v3io_dir_to_mount)
            source_to_copy = f"./{base.FETCHED_SOURCE_SUBDIR}/{basename}"
            return source_to_copy, None, v3io_dir_to_mount

        if "://" in source:
            # fail fast, before scheduling a pod, on a scheme `mlrun load-source` can't resolve
            # (the fetch-source init container would otherwise fail only once it actually runs).
            # git's own ref-fragment normalization happens inside `clone_git`, which
            # ``load-source`` calls - the raw source is handed through unchanged.
            if not mlrun.utils.clones.is_source_loadable(source):
                scheme = urlparse(source).scheme
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Source scheme '{scheme}://' is not supported by mlrun load-source. "
                    "Provide a store:// URI, a git:// URL, a .zip/.tar.gz archive, or a bare "
                    "s3:// / http(s):// file"
                )
            return f"./{base.FETCHED_SOURCE_SUBDIR}", source, None

        # no scheme: local path, same edge-case semantics Kaniko has - assumed to be a valid path
        # already inside the build image (e.g. baked into a custom base image); relative paths are
        # not supported at build time.
        if os.path.isabs(source):
            return source, None, None

        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Load of relative source ({source}) is not supported at build time "
            "see 'mlrun.runtimes.kubejob.KubejobRuntime.with_source_archive' or "
            "'mlrun.projects.project.MlrunProject.set_source' for more details"
        )


def make_buildah_pod(
    project: str,
    dest: str,
    dockerfile: str,
    inline_code: str | None = None,
    inline_path: str | None = None,
    requirements: list | None = None,
    requirements_path: str | None = None,
    secret_name: str | None = None,
    name: str = "",
    verbose: bool = False,
    builder_env: list | None = None,
    project_secrets: list | None = None,
    runtime_spec=None,
    registry: str | None = None,
    extra_labels: dict | None = None,
    project_default_function_node_selector: dict | None = None,
    auth_info: mlrun.common.schemas.AuthInfo | None = None,
) -> framework.utils.singletons.k8s.BasePod:
    """Construct the rootless Buildah build pod (``buildah bud`` + ``push``).

    :param project:            The project the build belongs to.
    :param dest:               The fully resolved destination image reference.
    :param dockerfile:         The rendered Dockerfile content (from :func:`base.make_dockerfile`).
    :param inline_code:        Inline function code to stage into the build context, if any.
    :param inline_path:        Destination filename for ``inline_code`` (default ``main.py``).
    :param requirements:       The resolved requirements list to stage, if any.
    :param requirements_path:  Path of the requirements file inside the build context.
    :param secret_name:        The docker-config secret used as the push authfile, if any.
    :param name:               The build pod's base name.
    :param verbose:            Whether to run Buildah with ``--log-level debug``.
    :param builder_env:        Build-time env vars, rendered as ``--build-arg`` (value read from env).
    :param project_secrets:    Project secrets, rendered as ``--build-arg`` (value read from env).
    :param runtime_spec:       The function's runtime spec (scheduling attributes, resources).
    :param registry:           The target registry, when given explicitly.
    :param extra_labels:       mlrun-internal labels to stamp on the build pod.
    :param project_default_function_node_selector: Project-level default node selector.
    :param auth_info:          The caller's auth info (for service-account resolution).
    :return: The Buildah build pod.
    """
    storage_driver = config.httpdb.builder.buildah_storage_driver
    if storage_driver not in _SUPPORTED_STORAGE_DRIVERS:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Unsupported buildah_storage_driver '{storage_driver}'. "
            f"Supported drivers: {', '.join(_SUPPORTED_STORAGE_DRIVERS)}"
        )

    if not registry:
        # if the registry was not given, infer it from the image destination
        registry = dest.partition("/")[0]

    # runtime-derived scheduling/identity pod-spec attributes (node selector, affinity, tolerations,
    # preemption, priority class, service account), resolved by the shared helper both backends call.
    extra_runtime_spec = base.resolve_build_pod_spec_attributes(
        project,
        runtime_spec,
        project_default_function_node_selector,
        auth_info,
    )

    # stage the Dockerfile (and any inline code / requirements) into the context, then bud + push.
    # everything runs in one container: the buildah image ships bash + coreutils, so no separate
    # dockerfile-creation init container is needed (unlike Kaniko).
    env = _build_env(
        dockerfile=dockerfile,
        inline_code=inline_code,
        requirements=requirements,
        requirements_path=requirements_path,
        secret_name=secret_name,
        builder_env=builder_env,
        project_secrets=project_secrets,
    )
    script = _build_script(
        dest=dest,
        storage_driver=storage_driver,
        verbose=verbose,
        secret_name=secret_name,
        inline_code=inline_code,
        inline_path=inline_path,
        requirements=requirements,
        requirements_path=requirements_path,
        build_arg_names=[env_var.name for env_var in (builder_env or [])]
        + [env_var.name for env_var in (project_secrets or [])],
    )

    buildah_pod = framework.utils.singletons.k8s.BasePod(
        name or "mlrun-build",
        config.httpdb.builder.buildah_image,
        command=["/bin/bash", "-c"],
        args=[script],
        kind="build",
        project=project,
        default_pod_spec_attributes=extra_runtime_spec,
        resources=base.build_pod_resources(runtime_spec),
        labels=base.resolve_builder_pod_labels(extra_labels),
        security_context=_caps_security_context(),
        env=env,
    )

    # the caps rootless model needs the runtime's default AppArmor profile relaxed (it blocks the
    # mount/unshare syscalls the build performs); the annotation is keyed to the build container name.
    apparmor_profile = config.httpdb.builder.buildah_apparmor_profile
    if apparmor_profile:
        buildah_pod.add_annotation(
            f"{_APPARMOR_ANNOTATION_PREFIX}/{buildah_pod.container_name}",
            apparmor_profile,
        )

    # a writable containers store (overlay/vfs data) off the read-only image layers.
    buildah_pod.mount_empty(name="context", mount_path=_CONTEXT_DIR)
    buildah_pod.mount_empty(name="containers", mount_path=_CONTAINERS_STORE)

    if config.is_pip_ca_configured():
        _mount_pip_ca(buildah_pod)

    if secret_name:
        # mount the docker-config secret as the push authfile (static-credential registries).
        buildah_pod.mount_secret(
            secret_name,
            _AUTHFILE_DIR,
            items=[{"key": ".dockerconfigjson", "path": "config.json"}],
        )

    mlrun.utils.logger.debug(
        "Resolved buildah build pod",
        project=project,
        image=dest,
        storage_driver=storage_driver,
        apparmor_profile=apparmor_profile or None,
    )
    return buildah_pod


def _caps_security_context() -> client.V1SecurityContext:
    # the caps rootless model (POC-1-validated): run as the image's non-root build user and add only
    # SETUID/SETGID (+ allowPrivilegeEscalation) so buildah can run newuidmap/newgidmap to set up its
    # subuid/subgid mapping. Not privileged. The hostUsers model was dropped (not viable per POC-1).
    return client.V1SecurityContext(
        run_as_user=_BUILD_UID,
        run_as_group=_BUILD_GID,
        run_as_non_root=True,
        allow_privilege_escalation=True,
        privileged=False,
        capabilities=client.V1Capabilities(add=["SETUID", "SETGID"]),
    )


def _build_env(
    dockerfile: str,
    inline_code: str | None,
    requirements: list | None,
    requirements_path: str | None,
    secret_name: str | None,
    builder_env: list | None,
    project_secrets: list | None,
) -> list[client.V1EnvVar]:
    # base64 the staged files into env vars (decoded in the container) to avoid shell-quoting the
    # contents into the build script - the same convention Kaniko's create-dockerfile init uses.
    env = [
        client.V1EnvVar(name="BUILDAH_ISOLATION", value="chroot"),
        client.V1EnvVar(name="HOME", value=_BUILD_HOME),
        client.V1EnvVar(name="MLRUN_DOCKERFILE", value=_b64(dockerfile)),
    ]
    if secret_name:
        env.append(client.V1EnvVar(name="REGISTRY_AUTH_FILE", value=_AUTHFILE_PATH))
    if inline_code:
        env.append(client.V1EnvVar(name="MLRUN_INLINE_CODE", value=_b64(inline_code)))
    # gate on the same condition _build_script decodes it (requirements need a target path), so the
    # env var and the decode stay in lock-step - no dead env var, no un-staged requirements.
    if requirements and requirements_path:
        requirements_content = "{}\n".format("\n".join(requirements))
        env.append(
            client.V1EnvVar(name="MLRUN_REQUIREMENTS", value=_b64(requirements_content))
        )
    # build-args are referenced by name only in the script; their values are read from the pod env,
    # so the plain builder-env values and the secret-backed (valueFrom) vars must both be present.
    env += list(builder_env or []) + list(project_secrets or [])
    return env


def _build_script(
    dest: str,
    storage_driver: str,
    verbose: bool,
    secret_name: str | None,
    inline_code: str | None,
    inline_path: str | None,
    requirements: list | None,
    requirements_path: str | None,
    build_arg_names: list[str],
) -> str:
    dockerfile_path = f"{_CONTEXT_DIR}/Dockerfile"
    lines = [
        "set -e",
        f"echo ${{MLRUN_DOCKERFILE}} | base64 -d > {shlex.quote(dockerfile_path)}",
    ]
    if inline_code:
        inline_target = f"{_CONTEXT_DIR}/{inline_path or 'main.py'}"
        lines.append(
            f"echo ${{MLRUN_INLINE_CODE}} | base64 -d > {shlex.quote(inline_target)}"
        )
    if requirements and requirements_path:
        lines.append(
            f"echo ${{MLRUN_REQUIREMENTS}} | base64 -d > {shlex.quote(requirements_path)}"
        )

    # buildah global options precede the subcommand.
    global_opts = ["buildah"]
    if verbose:
        global_opts += ["--log-level", "debug"]
    global_opts += ["--storage-driver", storage_driver]

    bud = global_opts + ["bud"]
    bud += _tls_verify_flag(
        config.httpdb.builder.insecure_pull_registry_mode, secret_name
    )
    for arg_name in build_arg_names:
        bud += ["--build-arg", arg_name]
    bud += ["--tag", dest, "--file", dockerfile_path, _CONTEXT_DIR]

    push = global_opts + ["push"]
    push += _tls_verify_flag(
        config.httpdb.builder.insecure_push_registry_mode, secret_name
    )
    push += ["--retry", str(config.httpdb.builder.buildah_push_retry)]
    if secret_name:
        push += ["--authfile", _AUTHFILE_PATH]
    push += [dest, f"docker://{dest}"]

    lines.append(shlex.join(bud))
    lines.append(shlex.join(push))
    return "\n".join(lines)


def _tls_verify_flag(mode: str, secret_name: str | None) -> list[str]:
    # mirror Kaniko's insecure-registry resolution: "enabled" always, "auto" when there is no
    # docker-config secret, "disabled" never. Buildah expresses it as --tls-verify=false.
    if mode == "enabled" or (mode == "auto" and not secret_name):
        return ["--tls-verify=false"]
    return []


def _mount_pip_ca(buildah_pod: framework.utils.singletons.k8s.BasePod) -> None:
    # mirror Kaniko: stage the pip CA cert into the build context so the Dockerfile's COPY (added by
    # base.make_dockerfile when a pip CA is configured) resolves.
    ca_filename = pathlib.Path(config.httpdb.builder.pip_ca_path).name
    buildah_pod.mount_secret(
        config.httpdb.builder.pip_ca_secret_name,
        str(pathlib.Path(_CONTEXT_DIR) / ca_filename),
        items=[{"key": config.httpdb.builder.pip_ca_secret_key, "path": ca_filename}],
        sub_path=ca_filename,
    )


def _b64(content: str) -> str:
    return b64encode(content.encode("utf-8")).decode("utf-8")
