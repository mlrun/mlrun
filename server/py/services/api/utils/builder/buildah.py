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
import pathlib
import shlex
from base64 import b64encode

from kubernetes import client

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils
from mlrun.config import config

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

# the push authfile: either the mounted static docker-config secret, or - for ECR/ACR - written by
# the registry_auth init container onto this same (emptyDir) path; GAR writes it here too, but
# just-in-time in this container's own push script (see registry_auth for why). See
# services.api.utils.builder.registry_auth for the cloud-credential-exchange implementations.
_AUTHFILE_DIR = "/auth"
_AUTHFILE_PATH = f"{_AUTHFILE_DIR}/config.json"

# where the GAR/GCR JIT credential-exchange script (see registry_auth.gar_credential_exchange_script)
# is decoded to before running it.
_GAR_SCRIPT_PATH = "/tmp/mlrun-gar-credential-exchange.py"

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

    Scope note: this adapter handles the no-source-context build (inline code / requirements /
    commands). Registry auth covers static docker-config secrets and ECR/ACR/GAR credential
    exchange (see :mod:`services.api.utils.builder.registry_auth`); remote source acquisition
    (ML-12887) is not implemented here - :func:`~services.api.utils.builder.resolve_builder_backend`
    transparently falls back to Kaniko for that input until its follow-up lands.
    """

    def make_build_pod(
        self, request: base.BuildRequest
    ) -> framework.utils.singletons.k8s.BasePod:
        """Build the rootless Buildah build pod for ``request``.

        :param request: The resolved, engine-agnostic build inputs.
        :return: The Buildah build pod.
        :raises mlrun.errors.MLRunInvalidArgumentError: If the request carries a source needing
            acquisition (resolve_builder_backend should have routed it to Kaniko - ML-12887).
        """
        # source acquisition is not implemented in this adapter yet (ML-12887). resolve_builder_backend
        # falls back to Kaniko when a request carries such a source, so reaching here with one is a bug
        # in the selection gate, not a user error - fail fast rather than silently drop the source.
        loads_source_on_run = bool(
            request.runtime_spec and request.runtime_spec.build.load_source_on_run
        )
        if request.source and not (request.inline_code or loads_source_on_run):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "BuildahBackend received a build source it cannot acquire yet (ML-12887); "
                "this should have fallen back to Kaniko in resolve_builder_backend"
            )
        dockerfile = base.make_dockerfile(
            request.base_image,
            request.commands,
            source=None,
            requirements_path=request.requirements_path,
            extra=request.extra,
            builder_env=request.builder_env_list,
            project_secrets=request.project_secrets,
            extra_args=request.extra_args,
        )
        return make_buildah_pod(
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

    # ECR/ACR/GAR need credential-exchange auth (see registry_auth); anything else (Docker Hub,
    # private, self-signed) sticks to the static docker-config secret path, unchanged. An explicit
    # secret_name always wins - e.g. a self-hosted credential on an otherwise-cloud registry host -
    # so it must never be overridden by an inferred cloud-provider exchange.
    cloud_provider = (
        None if secret_name else registry_auth.classify_cloud_registry(registry)
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
        cloud_provider=cloud_provider,
        registry=registry,
        builder_env=builder_env,
        project_secrets=project_secrets,
    )
    script = _build_script(
        dest=dest,
        storage_driver=storage_driver,
        verbose=verbose,
        secret_name=secret_name,
        cloud_provider=cloud_provider,
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
    elif cloud_provider in ("ecr", "acr"):
        # shared with the credential-exchange init container below - it writes the authfile here,
        # this container reads it at push time.
        buildah_pod.mount_empty(name="registry-auth", mount_path=_AUTHFILE_DIR)
        if cloud_provider == "ecr":
            registry_auth.append_ecr_credential_exchange_init_container(
                buildah_pod, registry, dest, _AUTHFILE_PATH
            )
        else:
            registry_auth.append_acr_credential_exchange_init_container(
                buildah_pod, registry, _AUTHFILE_PATH
            )
    # GAR/GCR needs no mount or init container - the authfile is written just-in-time by this
    # container's own push script (see _build_script), directly into its writable root filesystem.

    mlrun.utils.logger.debug(
        "Resolved buildah build pod",
        project=project,
        image=dest,
        storage_driver=storage_driver,
        cloud_provider=cloud_provider,
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
    cloud_provider: str | None,
    registry: str,
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
    if secret_name or cloud_provider:
        env.append(client.V1EnvVar(name="REGISTRY_AUTH_FILE", value=_AUTHFILE_PATH))
    if cloud_provider == "gar":
        # minted just-in-time by this same container's push script, not an init container - see
        # registry_auth.gar_credential_exchange_script.
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
    cloud_provider: str | None,
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

    has_push_auth = bool(secret_name or cloud_provider)

    # GAR/GCR credentials are minted JIT (see registry_auth.gar_credential_exchange_script) rather
    # than by an earlier init container, since GCP metadata-server tokens default to a 1h TTL that a
    # long build could outlive. Minted immediately before both bud (in case the base image shares
    # the same registry) and push, rather than once up front, for the same TTL reason.
    gar_credential_exchange = (
        [
            f"mkdir -p {shlex.quote(_AUTHFILE_DIR)}",
            f"echo ${{MLRUN_GAR_CREDENTIAL_SCRIPT}} | base64 -d > {shlex.quote(_GAR_SCRIPT_PATH)}",
            shlex.join(["python3", _GAR_SCRIPT_PATH]),
        ]
        if cloud_provider == "gar"
        else []
    )
    lines += gar_credential_exchange

    bud = global_opts + ["bud"]
    # pull auth is intentionally scoped to secret_name, not cloud_provider: it's about arbitrary
    # base-image registries, unrelated to the *destination*'s cloud classification.
    bud += _tls_verify_flag(
        config.httpdb.builder.insecure_pull_registry_mode, bool(secret_name)
    )
    for arg_name in build_arg_names:
        bud += ["--build-arg", arg_name]
    bud += ["--tag", dest, "--file", dockerfile_path, _CONTEXT_DIR]
    lines.append(shlex.join(bud))

    lines += gar_credential_exchange

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
