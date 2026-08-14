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
from mlrun.utils.registry_auth import CloudRegistryProvider

import framework.utils.singletons.k8s
from services.api.utils.builder import base, registry_auth

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

# the authfile: an emptyDir populated by any combination of a copied-in static docker-config
# secret, ECR/ACR init containers, and GAR's JIT script in this container's own bud/push script -
# merged per registry (ML-12988), not mutually exclusive, so a secret authenticating one registry
# (e.g. a private base image) coexists with cloud exchange for another (e.g. the push
# destination). See services.api.utils.builder.registry_auth for the cloud-credential-exchange
# implementations.
_AUTHFILE_DIR = "/auth"
_AUTHFILE_PATH = f"{_AUTHFILE_DIR}/config.json"

# where the GAR/GCR JIT credential-exchange script (see registry_auth.gar_credential_exchange_script)
# is decoded to before running it - push destination and base image, minted independently.
_GAR_SCRIPT_PATH = "/tmp/mlrun-gar-credential-exchange.py"
_GAR_PULL_SCRIPT_PATH = "/tmp/mlrun-gar-credential-exchange-pull.py"

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

    Registry auth covers static docker-config secrets and ECR/ACR/GAR credential exchange (see
    :mod:`services.api.utils.builder.registry_auth`).

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
            base_image=request.base_image,
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
    base_image: str | None = None,
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
    :param base_image:         The image the Dockerfile builds ``FROM``, used to also credential
                               the base-image pull when it's on a cloud registry (ML-12961).
    :param inline_code:        Inline function code to stage into the build context, if any.
    :param inline_path:        Destination filename for ``inline_code`` (default ``main.py``).
    :param requirements:       The resolved requirements list to stage, if any.
    :param requirements_path:  Path of the requirements file inside the build context.
    :param secret_name:        A docker-config secret to merge into the authfile, if any - covers
                               whichever registries it has entries for; a cloud provider covering a
                               *different* registry still gets credential exchange (ML-12988).
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

    base_registry = base_image.partition("/")[0] if base_image else None

    # every distinct registry this build touches, and which role(s) it plays there - a secret may
    # authenticate a *different* registry than either of these (e.g. a private base-image registry
    # alongside a cloud push destination, ML-12988), so it's resolved independently below and
    # merged into the authfile instead (see the secret-copy init container), never gated by this.
    roles_by_registry: dict[str, set[str]] = {registry: {"push"}}
    if base_registry:
        roles_by_registry.setdefault(base_registry, set()).add("pull")

    # ECR/ACR/GAR need credential-exchange auth (see registry_auth); anything else (Docker Hub,
    # private, self-signed) sticks to the static docker-config secret path.
    push_cloud_provider = registry_auth.classify_cloud_registry(registry)
    # the base image's registry needs its own credential exchange too (ML-12961) - e.g. a "system"
    # ACR hosting mlrun's own images vs. a per-project push target. Classified unconditionally, even
    # when it's the same registry as the push destination: GAR mints pull and push credentials
    # independently either way (see _build_script); ECR/ACR instead dedupe on the shared registry
    # via roles_by_registry below.
    pull_cloud_provider = (
        registry_auth.classify_cloud_registry(base_registry) if base_registry else None
    )

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
        roles_by_registry=roles_by_registry,
        builder_env=builder_env,
        project_secrets=project_secrets,
    )
    script = _build_script(
        dest=dest,
        storage_driver=storage_driver,
        verbose=verbose,
        secret_name=secret_name,
        roles_by_registry=roles_by_registry,
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

    if push_cloud_provider or pull_cloud_provider:
        # an emptyDir, not the container's own root filesystem: confirmed on a live GKE cluster
        # that the rootless build user can't mkdir at the image's filesystem root ("/auth: Permission
        # denied") - the mount is what actually guarantees a writable path, regardless of provider.
        # For ECR/ACR it's also how the init container(s) below hand the authfile to this container.
        buildah_pod.mount_empty(name="registry-auth", mount_path=_AUTHFILE_DIR)
        if secret_name:
            # mounted read-only elsewhere, then copied in as the first init container (ML-12988) -
            # see registry_auth.append_secret_authfile_init_container.
            buildah_pod.mount_secret(
                secret_name,
                registry_auth.SECRET_AUTHFILE_DIR,
                items=[{"key": ".dockerconfigjson", "path": "config.json"}],
            )
            registry_auth.append_secret_authfile_init_container(
                buildah_pod, _AUTHFILE_PATH
            )
        # GAR/GCR gets no init container - the authfile is written just-in-time by this container's
        # own script (see _build_script), into the mount above rather than the root filesystem, for
        # both the push destination and the base image's registry.
        #
        # ECR/ACR mint via one init container per distinct registry: a registry playing both push
        # and pull (i.e. the push destination and base image are the same) gets exactly one,
        # parameterized by the union of its roles - only a push role creates the destination
        # repository, and only a pull-only registry (never a push-covering one) gets the "-pull"
        # container name, since a push-side container can coexist with a different pull-side
        # registry in the same pod.
        for target_registry, roles in roles_by_registry.items():
            provider = registry_auth.classify_cloud_registry(target_registry)
            container_name_kwargs = (
                {
                    "container_name": registry_auth.PULL_CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME
                }
                if roles == {"pull"}
                else {}
            )
            if provider == CloudRegistryProvider.ECR:
                registry_auth.append_ecr_credential_exchange_init_container(
                    buildah_pod,
                    target_registry,
                    _AUTHFILE_PATH,
                    dest=dest if "push" in roles else None,
                    **container_name_kwargs,
                )
            elif provider == CloudRegistryProvider.ACR:
                registry_auth.append_acr_credential_exchange_init_container(
                    buildah_pod,
                    target_registry,
                    _AUTHFILE_PATH,
                    **container_name_kwargs,
                )
    elif secret_name:
        # no cloud provider involved - mount the secret directly as the authfile, unchanged.
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
        push_cloud_provider=push_cloud_provider,
        pull_cloud_provider=pull_cloud_provider,
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
    roles_by_registry: dict[str, set[str]],
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
    has_cloud_registry = any(
        registry_auth.classify_cloud_registry(registry)
        for registry in roles_by_registry
    )
    if secret_name or has_cloud_registry:
        env.append(client.V1EnvVar(name="REGISTRY_AUTH_FILE", value=_AUTHFILE_PATH))
    # GAR/GCR is minted just-in-time by this same container's own script (see
    # registry_auth.gar_credential_exchange_script), never via an init container - a registry
    # playing both push and pull still gets both scripts independently, unlike ECR/ACR's init
    # containers above: see _build_script for why this one never dedupes on a shared registry.
    for registry, roles in roles_by_registry.items():
        if registry_auth.classify_cloud_registry(registry) != CloudRegistryProvider.GAR:
            continue
        if "push" in roles:
            env.append(
                client.V1EnvVar(
                    name="MLRUN_GAR_CREDENTIAL_SCRIPT",
                    value=_b64(
                        registry_auth.gar_credential_exchange_script(
                            registry, _AUTHFILE_PATH
                        )
                    ),
                )
            )
        if "pull" in roles:
            env.append(
                client.V1EnvVar(
                    name="MLRUN_GAR_PULL_CREDENTIAL_SCRIPT",
                    value=_b64(
                        registry_auth.gar_credential_exchange_script(
                            registry, _AUTHFILE_PATH
                        )
                    ),
                )
            )
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
    roles_by_registry: dict[str, set[str]],
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

    # `bud`/`push` are each a single Buildah invocation, so there's exactly one pull-side and one
    # push-side question here no matter how many registries roles_by_registry ever grows to hold -
    # this assumes one registry per role, true for both roles today (see _role_cloud_provider).
    push_cloud_provider = _role_cloud_provider(roles_by_registry, "push")
    pull_cloud_provider = _role_cloud_provider(roles_by_registry, "pull")

    has_push_auth = bool(secret_name or push_cloud_provider)
    # a static secret covers any registry; otherwise a dedicated pull-side exchange (ML-12961) covers
    # the base image's own registry - classified independently of push_cloud_provider (see
    # make_buildah_pod), so an unrelated push-side provider never wrongly asserts pull auth for a
    # self-signed/private base image and disables --tls-verify=false for it in "auto" mode.
    has_pull_auth = bool(secret_name or pull_cloud_provider)

    # GAR/GCR credentials are minted JIT (see registry_auth.gar_credential_exchange_script) rather
    # than by an earlier init container: the metadata server caches and reuses a token across callers
    # until fewer than 5 minutes remain before its expiry, so any mint can hand back a token with as
    # little as ~5 minutes left, regardless of when it's minted. Pull and push are minted
    # independently, each immediately before the one step that needs it, to minimize that gap.
    if push_cloud_provider == CloudRegistryProvider.GAR:
        push_gar_credential_exchange = [
            f"echo ${{MLRUN_GAR_CREDENTIAL_SCRIPT}} | base64 -d > {shlex.quote(_GAR_SCRIPT_PATH)}",
            registry_auth.soft_fail_script(["python3", _GAR_SCRIPT_PATH], "GAR"),
        ]
    else:
        push_gar_credential_exchange = []

    # the base image's own GAR registry (ML-12961) - only needed before `bud`, since that's the
    # only step that pulls the base image.
    if pull_cloud_provider == CloudRegistryProvider.GAR:
        pull_gar_credential_exchange = [
            f"echo ${{MLRUN_GAR_PULL_CREDENTIAL_SCRIPT}} | base64 -d > {shlex.quote(_GAR_PULL_SCRIPT_PATH)}",
            registry_auth.soft_fail_script(["python3", _GAR_PULL_SCRIPT_PATH], "GAR"),
        ]
    else:
        pull_gar_credential_exchange = []

    lines += pull_gar_credential_exchange

    bud = global_opts + ["bud"]
    # unlike Kaniko - whose RUN steps execute directly on the build pod's own filesystem - Buildah's
    # RUN steps run in a real, isolated build sandbox that can't see files staged into the context
    # dir (e.g. requirements.txt) unless it's bind-mounted back in.
    bud += ["--volume", f"{_CONTEXT_DIR}:{_CONTEXT_DIR}"]
    bud += _tls_verify_flag(
        config.httpdb.builder.insecure_pull_registry_mode, has_pull_auth
    )
    for arg_name in build_arg_names:
        bud += ["--build-arg", arg_name]
    bud += ["--tag", dest, "--file", dockerfile_path, _CONTEXT_DIR]
    lines.append(shlex.join(bud))

    lines += push_gar_credential_exchange

    push = global_opts + ["push"]
    push += _tls_verify_flag(
        config.httpdb.builder.insecure_push_registry_mode, has_push_auth
    )
    push += ["--retry", str(config.httpdb.builder.buildah_push_retry)]
    if has_push_auth:
        push += ["--authfile", _AUTHFILE_PATH]
    push += [dest, f"docker://{dest}"]
    lines.append(shlex.join(push))

    return "\n".join(lines)


def _role_cloud_provider(
    roles_by_registry: dict[str, set[str]], role: str
) -> CloudRegistryProvider | None:
    # today's build shape has exactly one registry per role (the push destination, the base image)
    # - if roles_by_registry ever grows a second registry for the same role (e.g. a multi-stage
    # build with several base images), this - and how has_pull_auth/has_push_auth aggregate it -
    # needs revisiting.
    registry = next(
        (r for r, roles in roles_by_registry.items() if role in roles), None
    )
    return registry_auth.classify_cloud_registry(registry) if registry else None


def _tls_verify_flag(mode: str, has_registry_auth: bool) -> list[str]:
    # mirror Kaniko's insecure-registry resolution: "enabled" always, "auto" when there is no
    # registry auth configured, "disabled" never. Buildah expresses it as --tls-verify=false.
    if mode == "enabled" or (mode == "auto" and not has_registry_auth):
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
