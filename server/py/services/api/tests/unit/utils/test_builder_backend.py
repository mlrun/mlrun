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

# Tests for the pluggable builder seam (ML-12884): the BuilderBackend selection
# gate and the BuildRequest contract. The behaviour of the Kaniko path itself is
# the byte-identical regression anchor covered by test_builder.py; here we lock the
# seam, and (ML-12885) the Buildah backend: config-flip selection, ECR/ACR/GAR
# credential exchange (ML-12886) - including the base-image (pull-side) exchange
# added in ML-12961 - remote source acquisition (ML-12887), and the rootless pod
# spec. It also locks Buildah's own source routing - every remote source is
# fetched via `mlrun load-source` or, for v3io, FUSE-mounted, since Buildah has no
# native remote-context resolution the way Kaniko does.

import base64
import unittest.mock

import pytest
from kubernetes import client

import mlrun
import mlrun.common.schemas
import mlrun.errors
from mlrun.config import config
from mlrun.runtimes import RuntimeKinds

import framework.utils.singletons.k8s
import services.api.utils.builder
import services.api.utils.builder.buildah


def test_resolve_builder_backend_default_is_kaniko():
    # the shipped default: no config change routes to the Kaniko adapter
    backend = services.api.utils.builder.resolve_builder_backend(_make_build_request())
    assert isinstance(backend, services.api.utils.builder.KanikoBackend)


def test_resolve_builder_backend_unknown_raises(monkeypatch):
    # a misconfigured / not-yet-registered backend fails fast with a clear error,
    # naming the offending value - rather than silently falling back to Kaniko
    monkeypatch.setattr(config.httpdb.builder, "builder_backend", "no-such-backend")
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match="no-such-backend"):
        services.api.utils.builder.resolve_builder_backend(_make_build_request())


def test_build_image_build_request_carries_raw_unrouted_source(monkeypatch):
    # the D1 seam contract: build_image hands the backend the *raw* source
    # descriptor, and routing it to a build context is engine-owned (below the
    # seam). A git source is a good probe - the backend rewrites the branch
    # fragment to refs/heads while routing, so the request must still hold the
    # untouched original.
    raw_source = "git://github.com/some-org/some-repo.git#main"

    captured = {}

    def _capture_request(self, request):
        captured["request"] = request
        return unittest.mock.MagicMock()

    monkeypatch.setattr(
        services.api.utils.builder.KanikoBackend, "make_build_pod", _capture_request
    )
    monkeypatch.setattr(
        config.httpdb.builder,
        "docker_registry",
        "default.docker.registry/default-repository",
    )
    _patch_k8s_helper(monkeypatch)

    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind=RuntimeKinds.job,
    )
    function.spec.build.source = raw_source
    services.api.utils.builder.build_runtime(
        mlrun.common.schemas.AuthInfo(),
        function,
    )

    request = captured["request"]
    assert request.source == raw_source
    # not routed: neither an empty/local context nor a branch-rewritten git ref
    assert request.source not in ("/empty", "/context", ".")
    assert "refs/heads" not in request.source


def _patch_k8s_helper(monkeypatch):
    get_k8s_helper_mock = unittest.mock.Mock()
    get_k8s_helper_mock.create_pod = unittest.mock.Mock(
        side_effect=lambda pod: (pod, "some-namespace")
    )
    get_k8s_helper_mock.get_project_secret_name = unittest.mock.Mock(
        side_effect=lambda name: "name"
    )
    get_k8s_helper_mock.get_project_secret_keys = unittest.mock.Mock(
        side_effect=lambda project, filter_internal: ["KEY"]
    )
    monkeypatch.setattr(
        framework.utils.singletons.k8s,
        "get_k8s_helper",
        lambda *args, **kwargs: get_k8s_helper_mock,
    )


def _make_build_request(**overrides) -> services.api.utils.builder.BuildRequest:
    """Build a minimal BuildRequest; override only the fields a test cares about."""
    defaults = dict(
        project="some-project",
        image_target="registry/some-image:tag",
        base_image="mlrun/mlrun",
        commands=[],
        requirements=[],
        requirements_path="",
        source="",
        inline_code=None,
        inline_path=None,
        extra=None,
        builder_env={},
        builder_env_list=[],
        project_secrets=[],
        extra_args={},
        secret_name=None,
        registry=None,
        runtime_spec=None,
        project_default_function_node_selector={},
        user_unix_id=None,
        enriched_group_id=None,
        auth_info=mlrun.common.schemas.AuthInfo(),
        name="mlrun-build",
        labels={},
        verbose=False,
    )
    defaults.update(overrides)
    return services.api.utils.builder.BuildRequest(**defaults)


# --- Buildah backend (ML-12885) ------------------------------------------------------------------


def test_resolve_builder_backend_buildah_selected(monkeypatch):
    # a buildah-configured cluster with a plain-registry, no-source build routes to the Buildah adapter
    monkeypatch.setattr(config.httpdb.builder, "builder_backend", "buildah")
    backend = services.api.utils.builder.resolve_builder_backend(
        _make_build_request(image_target="registry.example.com/some-image:tag")
    )
    assert isinstance(backend, services.api.utils.builder.BuildahBackend)


@pytest.mark.parametrize(
    "target",
    [
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/some-image:tag",  # ECR
        "myregistry.azurecr.io/some-image:tag",  # ACR
        "us-docker.pkg.dev/proj/repo/some-image:tag",  # GAR
    ],
)
def test_resolve_builder_backend_buildah_handles_cloud_registry_directly(
    monkeypatch, target
):
    # ECR/ACR/GAR credential exchange is wired into Buildah itself (ML-12886) -> no Kaniko fallback
    monkeypatch.setattr(config.httpdb.builder, "builder_backend", "buildah")
    backend = services.api.utils.builder.resolve_builder_backend(
        _make_build_request(image_target=target, registry=None)
    )
    assert isinstance(backend, services.api.utils.builder.BuildahBackend)


def test_resolve_builder_backend_buildah_handles_source(monkeypatch):
    # source acquisition is implemented -> Buildah is selected, no fallback
    monkeypatch.setattr(config.httpdb.builder, "builder_backend", "buildah")
    backend = services.api.utils.builder.resolve_builder_backend(
        _make_build_request(source="git://github.com/some-org/some-repo.git#main")
    )
    assert isinstance(backend, services.api.utils.builder.BuildahBackend)


def test_resolve_builder_backend_buildah_keeps_inline_code_with_source(monkeypatch):
    # inline_code doesn't stage a build-context source (Kaniko uses /empty too), so Buildah keeps it
    monkeypatch.setattr(config.httpdb.builder, "builder_backend", "buildah")
    backend = services.api.utils.builder.resolve_builder_backend(
        _make_build_request(
            source="git://github.com/some-org/some-repo.git#main",
            inline_code="print('hi')",
        )
    )
    assert isinstance(backend, services.api.utils.builder.BuildahBackend)


def test_make_buildah_pod_security_context_is_caps_rootless():
    # the caps rootless model (POC-1): non-root uid 1000 + SETUID/SETGID + allowPrivilegeEscalation,
    # never privileged. The hostUsers model was dropped, so host_users is never set on the pod.
    security_context = _make_buildah_pod().pod.spec.containers[0].security_context
    assert security_context.run_as_user == 1000
    assert security_context.run_as_group == 1000
    assert security_context.run_as_non_root is True
    assert security_context.allow_privilege_escalation is True
    assert security_context.privileged is False
    assert security_context.capabilities.add == ["SETUID", "SETGID"]


def test_make_buildah_pod_never_sets_host_users():
    # caps-only: the hostUsers rootless model was dropped (POC-1 showed it non-viable)
    assert getattr(_make_buildah_pod().pod.spec, "host_users", None) is None


@pytest.mark.parametrize("driver", ["overlay", "vfs"])
def test_make_buildah_pod_storage_driver_and_isolation(monkeypatch, driver):
    monkeypatch.setattr(config.httpdb.builder, "buildah_storage_driver", driver)
    buildah_pod = _make_buildah_pod()
    container = buildah_pod.pod.spec.containers[0]
    assert f"--storage-driver {driver}" in container.args[0]
    env = {env_var.name: env_var.value for env_var in container.env}
    assert env["BUILDAH_ISOLATION"] == "chroot"
    assert env["HOME"] == "/home/build"


def test_make_buildah_pod_invalid_storage_driver_raises(monkeypatch):
    monkeypatch.setattr(config.httpdb.builder, "buildah_storage_driver", "devicemapper")
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match="devicemapper"):
        _make_buildah_pod()


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("unconfined", "unconfined"),
        ("localhost/buildah", "localhost/buildah"),
        ("", None),
    ],
)
def test_make_buildah_pod_apparmor_annotation(monkeypatch, profile, expected):
    monkeypatch.setattr(config.httpdb.builder, "buildah_apparmor_profile", profile)
    pod = _make_buildah_pod().pod
    annotations = pod.metadata.annotations or {}
    prefix = "container.apparmor.security.beta.kubernetes.io"
    # the annotation must be keyed to the *actual* build container's name, otherwise k8s silently
    # ignores it (stale key -> default profile stays enforced -> GKE/AKS build fails with no clue).
    key = f"{prefix}/{pod.spec.containers[0].name}"
    if expected is None:
        # unset -> no apparmor annotation under any key
        assert not any(k.startswith(f"{prefix}/") for k in annotations)
    else:
        assert annotations[key] == expected


def test_make_buildah_pod_bud_and_push_args(monkeypatch):
    monkeypatch.setattr(config.httpdb.builder, "insecure_push_registry_mode", "enabled")
    monkeypatch.setattr(config.httpdb.builder, "insecure_pull_registry_mode", "enabled")
    monkeypatch.setattr(config.httpdb.builder, "buildah_push_retry", "5")
    script = _make_buildah_pod(verbose=True).pod.spec.containers[0].args[0]
    assert "buildah" in script and "bud" in script and "push" in script
    assert "--tls-verify=false" in script
    assert "--retry 5" in script
    assert "--log-level debug" in script
    assert "docker://docker-hub/some-image:tag" in script


def test_make_buildah_pod_tls_verify_disabled(monkeypatch):
    monkeypatch.setattr(
        config.httpdb.builder, "insecure_push_registry_mode", "disabled"
    )
    monkeypatch.setattr(
        config.httpdb.builder, "insecure_pull_registry_mode", "disabled"
    )
    script = _make_buildah_pod().pod.spec.containers[0].args[0]
    assert "--tls-verify=false" not in script


def test_make_buildah_pod_authfile_when_secret():
    buildah_pod = _make_buildah_pod(secret_name="my-docker-secret")
    container = buildah_pod.pod.spec.containers[0]
    assert "--authfile /auth/config.json" in container.args[0]
    env = {env_var.name: env_var.value for env_var in container.env}
    assert env["REGISTRY_AUTH_FILE"] == "/auth/config.json"
    mounted_secrets = {
        volume.secret.secret_name
        for volume in buildah_pod.pod.spec.volumes
        if volume.secret
    }
    assert "my-docker-secret" in mounted_secrets


@pytest.mark.parametrize(
    "registry,provider",
    [
        ("123456789012.dkr.ecr.us-east-1.amazonaws.com", "ecr"),
        ("myregistry.azurecr.io", "acr"),
        ("us-docker.pkg.dev", "gar"),
    ],
)
def test_make_buildah_pod_secret_and_cloud_provider_merge(registry, provider):
    # ML-12988: an explicit secret_name (e.g. authenticating a private base image elsewhere) must
    # not disable cloud credential exchange for a *different* host that needs it - the secret is
    # copied into the shared authfile first, then merged with per-host cloud auth on top.
    buildah_pod = _make_buildah_pod(
        dest=f"{registry}/some-image:tag",
        registry=registry,
        secret_name="my-docker-secret",
    )
    pod = buildah_pod.pod

    mounted_secrets = {
        volume.secret.secret_name for volume in pod.spec.volumes if volume.secret
    }
    assert mounted_secrets == {"my-docker-secret"}

    copy_container = _init_container_by_name(buildah_pod, "copy-registry-auth-secret")
    assert copy_container is not None
    assert copy_container.command == ["/bin/sh", "-c"]
    script = copy_container.args[0]
    assert "cp /auth-secret/config.json /auth/config.json" in script
    assert "chmod 0666 /auth/config.json" in script
    # mounted read-only elsewhere - never directly at the shared authfile path (ML-12988)
    secret_mount = next(
        mount
        for mount in copy_container.volume_mounts
        if mount.name == "my-docker-secret"
    )
    assert secret_mount.mount_path == "/auth-secret"
    # runs first, so cloud exchange below merges on top of the secret's own entries
    assert pod.spec.init_containers[0].name == "copy-registry-auth-secret"

    init_container_names = {c.name for c in pod.spec.init_containers}
    env = {env_var.name: env_var.value for env_var in pod.spec.containers[0].env}
    if provider == "gar":
        # GAR still has no credential-exchange init container - JIT in the main container
        assert init_container_names == {"copy-registry-auth-secret"}
        assert "MLRUN_GAR_CREDENTIAL_SCRIPT" in env
    else:
        assert init_container_names == {
            "copy-registry-auth-secret",
            "registry-credential-exchange",
        }


@pytest.mark.parametrize(
    "registry,provider",
    [
        ("123456789012.dkr.ecr.us-east-1.amazonaws.com", "ecr"),
        ("myregistry.azurecr.io", "acr"),
    ],
)
def test_make_buildah_pod_credential_exchange_init_container(registry, provider):
    # ECR/ACR: no secret_name, but the registry is a cloud one -> a shared authfile emptyDir plus a
    # credential-exchange init container, and the main container still gets --authfile/REGISTRY_AUTH_FILE
    buildah_pod = _make_buildah_pod(
        dest=f"{registry}/some-image:tag", registry=registry
    )
    pod = buildah_pod.pod

    init_containers = pod.spec.init_containers
    assert len(init_containers) == 1
    assert init_containers[0].name == "registry-credential-exchange"
    # same python -m mlrun <subcommand> convention Kaniko's source-fetch init container uses,
    # wrapped so a failed mint (ML-12988) logs a warning and exits 0 instead of failing the pod
    assert init_containers[0].command == ["/bin/sh", "-c"]
    script = init_containers[0].args[0]
    assert script.startswith("python -m mlrun mint-registry-credentials")
    assert f"--provider {provider}" in script
    assert "--authfile /auth/config.json" in script
    if provider == "ecr":
        assert f"--dest {registry}/some-image:tag" in script
    else:
        assert "--dest" not in script
    # the mounted emptyDir is shared: BasePod attaches every volume mount to every init container
    assert any(
        mount.mount_path == "/auth" for mount in init_containers[0].volume_mounts
    )

    empty_dir_volumes = {
        volume.name for volume in pod.spec.volumes if volume.empty_dir is not None
    }
    assert "registry-auth" in empty_dir_volumes

    container = pod.spec.containers[0]
    env = {env_var.name: env_var.value for env_var in container.env}
    assert env["REGISTRY_AUTH_FILE"] == "/auth/config.json"
    assert "--authfile /auth/config.json" in container.args[0]


def test_make_buildah_pod_gar_credential_exchange_is_jit_not_init_container():
    # GAR/GCR: minted just-in-time in the main container's push script, no init container - but it
    # still gets the registry-auth emptyDir mount, same as ECR/ACR: confirmed on a live GKE cluster
    # that the rootless build user can't mkdir on the image's own filesystem root ("permission
    # denied"), so the mount is what actually guarantees a writable authfile path.
    registry = "us-docker.pkg.dev"
    buildah_pod = _make_buildah_pod(
        dest=f"{registry}/proj/repo/some-image:tag", registry=registry
    )
    pod = buildah_pod.pod

    assert pod.spec.init_containers == []
    empty_dir_volumes = {
        volume.name for volume in pod.spec.volumes if volume.empty_dir is not None
    }
    assert "registry-auth" in empty_dir_volumes

    container = pod.spec.containers[0]
    env = {env_var.name: env_var.value for env_var in container.env}
    assert env["REGISTRY_AUTH_FILE"] == "/auth/config.json"
    assert "MLRUN_GAR_CREDENTIAL_SCRIPT" in env
    lines = container.args[0].splitlines()
    push_line = next(i for i, line in enumerate(lines) if "push" in line.split())
    mint_lines = [
        i for i, line in enumerate(lines) if "MLRUN_GAR_CREDENTIAL_SCRIPT" in line
    ]
    # minted once, immediately before push - no base image here, so bud needs no push credential.
    assert len(mint_lines) == 1
    assert mint_lines[0] < push_line
    mint_script_line = lines[push_line - 1]
    # wrapped so a failed mint (ML-12988) logs a warning and continues rather than aborting the
    # build under `set -e` - see registry_auth.soft_fail_script.
    assert mint_script_line.startswith("python3 /tmp/mlrun-gar-credential-exchange.py")
    assert mint_script_line.endswith(
        "|| echo 'WARNING: failed to mint GAR registry credentials' >&2"
    )
    assert "--authfile /auth/config.json" in lines[push_line]


def test_make_buildah_pod_bare_base_image_has_no_pull_side_exchange():
    # a base image with no explicit registry (e.g. the default "mlrun/mlrun:<version>", a Docker
    # Hub image) must not be treated as a pull-side registry needing credential exchange.
    registry = "myregistry.azurecr.io"
    buildah_pod = _make_buildah_pod(
        dest=f"{registry}/some-image:tag",
        registry=registry,
        base_image="mlrun/mlrun:1.13.0-rc2",
    )
    pod = buildah_pod.pod
    assert len(pod.spec.init_containers) == 1
    assert pod.spec.init_containers[0].name == "registry-credential-exchange"


def test_make_buildah_pod_bare_push_destination_has_no_push_side_exchange():
    # a push destination with no explicit registry (e.g. pushed straight to Docker Hub as
    # "my-org/my-image:tag") must not be treated as a registry needing credential exchange -
    # inferring one from the whole "my-org" segment would be as wrong as the base_image case above.
    base_registry = "myregistry.azurecr.io"
    buildah_pod = _make_buildah_pod(
        dest="my-org/my-image:tag",
        base_image=f"{base_registry}/mlrun/mlrun:1.13.0-rc2",
    )
    pod = buildah_pod.pod
    assert len(pod.spec.init_containers) == 1
    assert pod.spec.init_containers[0].name == "registry-credential-exchange-pull"


def test_make_buildah_pod_gar_credential_exchange_same_registry_mints_both():
    # base image on the *same* GAR host as the push destination - pull and push are still minted
    # independently (no same-host dedup for GAR, unlike ECR/ACR's init-container skip): pull right
    # before `bud`, push right before `push`.
    registry = "us-docker.pkg.dev"
    buildah_pod = _make_buildah_pod(
        dest=f"{registry}/proj/repo/some-image:tag",
        registry=registry,
        base_image=f"{registry}/mlrun/mlrun:1.13.0-rc2",
    )
    pod = buildah_pod.pod
    env = {env_var.name: env_var.value for env_var in pod.spec.containers[0].env}
    assert "MLRUN_GAR_CREDENTIAL_SCRIPT" in env
    assert "MLRUN_GAR_PULL_CREDENTIAL_SCRIPT" in env

    lines = pod.spec.containers[0].args[0].splitlines()
    bud_line = next(i for i, line in enumerate(lines) if "bud" in line.split())
    push_line = next(i for i, line in enumerate(lines) if "push" in line.split())
    pull_mint_line = next(
        i for i, line in enumerate(lines) if "MLRUN_GAR_PULL_CREDENTIAL_SCRIPT" in line
    )
    push_mint_line = next(
        i for i, line in enumerate(lines) if "MLRUN_GAR_CREDENTIAL_SCRIPT" in line
    )
    assert pull_mint_line < bud_line < push_mint_line < push_line


def test_make_buildah_pod_gar_pull_side_credential_exchange_for_different_host_base_image():
    # ML-12961: base image on a *different* GAR host than the push destination - a separate JIT
    # script mints its credentials, merged into the same authfile, run once right before `bud`
    # (the only step that pulls the base image).
    push_registry = "us-docker.pkg.dev"
    base_registry = "europe-docker.pkg.dev"
    buildah_pod = _make_buildah_pod(
        dest=f"{push_registry}/proj/repo/some-image:tag",
        registry=push_registry,
        base_image=f"{base_registry}/mlrun/mlrun:1.13.0-rc2",
    )
    pod = buildah_pod.pod

    assert pod.spec.init_containers == []
    container = pod.spec.containers[0]
    env = {env_var.name: env_var.value for env_var in container.env}
    assert "MLRUN_GAR_CREDENTIAL_SCRIPT" in env
    assert "MLRUN_GAR_PULL_CREDENTIAL_SCRIPT" in env

    lines = container.args[0].splitlines()
    bud_line = next(i for i, line in enumerate(lines) if "bud" in line.split())
    pull_mint_line = next(
        i for i, line in enumerate(lines) if "MLRUN_GAR_PULL_CREDENTIAL_SCRIPT" in line
    )
    assert pull_mint_line < bud_line
    pull_mint_script_line = lines[pull_mint_line + 1]
    assert pull_mint_script_line.startswith(
        "python3 /tmp/mlrun-gar-credential-exchange-pull.py"
    )
    assert pull_mint_script_line.endswith(
        "|| echo 'WARNING: failed to mint GAR registry credentials' >&2"
    )
    # runs exactly once - only needed before the base-image pull, not before push
    assert sum(1 for line in lines if "MLRUN_GAR_PULL_CREDENTIAL_SCRIPT" in line) == 1


@pytest.mark.parametrize(
    "base_registry,provider",
    [
        ("system.azurecr.io", "acr"),
        ("111111111111.dkr.ecr.us-east-1.amazonaws.com", "ecr"),
    ],
)
def test_make_buildah_pod_pull_side_credential_exchange_for_different_host_base_image(
    base_registry, provider
):
    # ML-12961 regression: the base image's registry needs its own credential exchange when it's a
    # different cloud host than the push destination - this is the reported "invalid
    # username/password" pulling a private ACR base image while pushing elsewhere.
    buildah_pod = _make_buildah_pod(
        dest="docker-hub/some-image:tag",
        base_image=f"{base_registry}/mlrun/mlrun:1.13.0-rc2",
    )
    pod = buildah_pod.pod

    init_containers = pod.spec.init_containers
    assert len(init_containers) == 1
    assert init_containers[0].name == "registry-credential-exchange-pull"
    script = init_containers[0].args[0]
    assert f"--provider {provider}" in script
    assert f"--registry {base_registry}" in script
    assert "--dest" not in script

    empty_dir_volumes = {
        volume.name for volume in pod.spec.volumes if volume.empty_dir is not None
    }
    assert "registry-auth" in empty_dir_volumes
    env = {env_var.name: env_var.value for env_var in pod.spec.containers[0].env}
    assert env["REGISTRY_AUTH_FILE"] == "/auth/config.json"


def test_make_buildah_pod_push_and_pull_different_acr_hosts():
    # push destination and base image are both ACR, but different registry instances - each needs
    # its own credential-exchange init container, writing into the same shared authfile.
    buildah_pod = _make_buildah_pod(
        dest="push.azurecr.io/some-image:tag",
        registry="push.azurecr.io",
        base_image="system.azurecr.io/mlrun/mlrun:1.13.0-rc2",
    )
    init_containers = buildah_pod.pod.spec.init_containers
    assert len(init_containers) == 2
    names = {container.name for container in init_containers}
    assert names == {
        "registry-credential-exchange",
        "registry-credential-exchange-pull",
    }

    by_name = {container.name: container for container in init_containers}
    assert "push.azurecr.io" in by_name["registry-credential-exchange"].args[0]
    assert "system.azurecr.io" in by_name["registry-credential-exchange-pull"].args[0]


@pytest.mark.parametrize(
    "registry",
    [
        "myregistry.azurecr.io",
        "123456789012.dkr.ecr.us-east-1.amazonaws.com",
    ],
)
def test_make_buildah_pod_same_registry_skips_duplicate_exchange(registry):
    # base image on the *same* ACR/ECR host as the push destination - the push-side exchange
    # already writes that host's authfile entry, so no second (redundant) init container is needed.
    buildah_pod = _make_buildah_pod(
        dest=f"{registry}/some-image:tag",
        registry=registry,
        base_image=f"{registry}/mlrun/mlrun:1.13.0-rc2",
    )
    init_containers = buildah_pod.pod.spec.init_containers
    assert len(init_containers) == 1
    assert init_containers[0].name == "registry-credential-exchange"

    # the pull *is* credentialed (via the init container above), even though it got no init
    # container of its own - --tls-verify=false must not be added for it in "auto" mode.
    script = buildah_pod.pod.spec.containers[0].args[0]
    bud_line = next(
        line
        for line in script.splitlines()
        if line.startswith("buildah") and " bud " in line
    )
    assert "--tls-verify=false" not in bud_line


def test_make_buildah_pod_secret_and_cloud_pull_exchange_merge():
    # ML-12988: secret_name authenticates the push destination (a non-cloud registry here), but
    # the base image lives on a recognized cloud registry - pull-side credential exchange must
    # still run, merging into the authfile alongside the copied-in secret.
    buildah_pod = _make_buildah_pod(
        dest="docker-hub/some-image:tag",
        base_image="system.azurecr.io/mlrun/mlrun:1.13.0-rc2",
        secret_name="my-docker-secret",
    )
    init_container_names = [
        container.name for container in buildah_pod.pod.spec.init_containers
    ]
    assert init_container_names == [
        "copy-registry-auth-secret",
        "registry-credential-exchange-pull",
    ]


def test_make_buildah_pod_pull_tls_verify_unaffected_by_unrelated_gar_push():
    # regression guard: a GAR push destination must not, on its own, mark the *pull* as
    # credentialed when the base image is on an unrelated (here: non-cloud/self-signed) registry -
    # doing so would wrongly disable --tls-verify=false for that pull in "auto" mode.
    registry = "us-docker.pkg.dev"
    script = (
        _make_buildah_pod(
            dest=f"{registry}/proj/repo/some-image:tag",
            registry=registry,
            base_image="self-signed.internal.example.com/base:latest",
        )
        .pod.spec.containers[0]
        .args[0]
    )
    bud_line = next(
        line
        for line in script.splitlines()
        if line.startswith("buildah") and " bud " in line
    )
    assert "--tls-verify=false" in bud_line


def test_make_buildah_pod_stages_inline_code_and_requirements():
    container = _make_buildah_pod(
        inline_code="print('hi')",
        inline_path="handler.py",
        requirements=["pandas==2.0.0"],
        requirements_path="/empty/requirements.txt",
    ).pod.spec.containers[0]
    env_names = {env_var.name for env_var in container.env}
    assert {"MLRUN_DOCKERFILE", "MLRUN_INLINE_CODE", "MLRUN_REQUIREMENTS"} <= env_names
    assert "/empty/handler.py" in container.args[0]
    assert "/empty/requirements.txt" in container.args[0]


def test_make_buildah_pod_bud_mounts_context_dir_for_run_steps():
    # Buildah's RUN steps execute in an isolated build sandbox, unlike Kaniko's (which share the
    # pod's own filesystem) - without this mount, a RUN referencing a context-staged file (e.g. the
    # requirements.txt written to /empty by this same script) can't see it.
    script = (
        _make_buildah_pod(
            requirements=["pandas==2.0.0"],
            requirements_path="/empty/requirements.txt",
        )
        .pod.spec.containers[0]
        .args[0]
    )
    bud_line = next(
        line
        for line in script.splitlines()
        if line.startswith("buildah") and " bud " in line
    )
    assert "--volume /empty:/empty" in bud_line


def test_make_buildah_pod_renders_build_args():
    container = _make_buildah_pod(
        builder_env=[client.V1EnvVar(name="ARG1", value="v1")],
        project_secrets=[client.V1EnvVar(name="SECRET1", value="s1")],
    ).pod.spec.containers[0]
    # build-args are referenced by name in the script; values are read from the pod env
    assert "--build-arg ARG1" in container.args[0]
    assert "--build-arg SECRET1" in container.args[0]
    assert {"ARG1", "SECRET1"} <= {env_var.name for env_var in container.env}


def test_base_pod_security_context_and_env_are_optional():
    # absent by default -> container security context / env unset (the Kaniko byte-identity anchor)
    pod = framework.utils.singletons.k8s.BasePod(
        task_name="t", image="img", default_pod_spec_attributes={}
    ).pod
    assert pod.spec.containers[0].security_context is None
    assert pod.spec.containers[0].env is None
    # when provided, they land on the container
    pod = framework.utils.singletons.k8s.BasePod(
        task_name="t",
        image="img",
        default_pod_spec_attributes={},
        security_context=client.V1SecurityContext(run_as_user=1000),
        env=[client.V1EnvVar(name="K", value="V")],
    ).pod
    assert pod.spec.containers[0].security_context.run_as_user == 1000
    assert pod.spec.containers[0].env[0].name == "K"


def _make_buildah_pod(**overrides) -> framework.utils.singletons.k8s.BasePod:
    """Build a Buildah pod with a real runtime spec; override only what a test cares about."""
    with unittest.mock.patch(
        "framework.api.utils.resolve_project_service_account_details",
        return_value=(None, None, None),
    ):
        function = mlrun.new_function("test", "test", kind=RuntimeKinds.job)
        defaults = dict(
            project="test",
            dest="docker-hub/some-image:tag",
            dockerfile="FROM alpine:3.20\n",
            runtime_spec=function.spec,
        )
        defaults.update(overrides)
        return services.api.utils.builder.buildah.make_buildah_pod(**defaults)


def _make_buildah_backend_pod(
    **overrides,
) -> framework.utils.singletons.k8s.BasePod:
    """Route a BuildRequest through BuildahBackend.make_build_pod with a real runtime spec.

    A real runtime_spec is required: source routing mutates
    ``request.runtime_spec.build.source_code_target_dir``, which a bare ``_make_build_request()``
    (``runtime_spec=None``) cannot support.
    """
    with unittest.mock.patch(
        "framework.api.utils.resolve_project_service_account_details",
        return_value=(None, None, None),
    ):
        function = mlrun.new_function("test", "test", kind=RuntimeKinds.job)
        request = _make_build_request(runtime_spec=function.spec, **overrides)
        return services.api.utils.builder.BuildahBackend().make_build_pod(request)


def _init_container_by_name(pod: framework.utils.singletons.k8s.BasePod, name: str):
    for container in pod.pod.spec.init_containers or []:
        if container.name == name:
            return container
    return None


def _decoded_dockerfile(pod: framework.utils.singletons.k8s.BasePod) -> str:
    env = {env_var.name: env_var.value for env_var in pod.pod.spec.containers[0].env}
    return base64.b64decode(env["MLRUN_DOCKERFILE"]).decode("utf-8")


@pytest.mark.parametrize(
    "source",
    [
        "git://github.com/some-org/some-repo.git#main",
        "s3://bucket/path/project.tar.gz",
        "s3://bucket/path/main.py",
        "http://example.com/main.py",
        "https://example.com/main.py",
    ],
)
def test_buildah_backend_routes_remote_source_through_fetch_init_container(source):
    # unlike Kaniko (native --context for git/s3, Dockerfile ADD for bare http), Buildah has no
    # native remote-context resolution - every remote source is fetched via `mlrun load-source`
    # into the same emptyDir already mounted for Dockerfile staging.
    pod = _make_buildah_backend_pod(source=source)

    fetch_container = _init_container_by_name(pod, "fetch-source")
    assert fetch_container is not None
    assert fetch_container.command == ["python"]
    assert fetch_container.args == [
        "-m",
        "mlrun",
        "load-source",
        source,
        "--target",
        "/empty/source",
    ]
    fetch_mounts = {
        (vm.name, vm.mount_path) for vm in fetch_container.volume_mounts or []
    }
    assert ("context", "/empty") in fetch_mounts

    assert "ADD ./source /home/mlrun_code" in _decoded_dockerfile(pod)


def test_buildah_backend_mounts_v3io_source():
    # v3io keeps its existing FUSE-mount mechanism (shared with Kaniko via
    # base.mount_v3io_source) rather than being routed through the fetch-source init container.
    pod = _make_buildah_backend_pod(
        source="v3io:///bigdata/project/code",
        auth_info=mlrun.common.schemas.AuthInfo(
            username="some-user", access_key="some-key"
        ),
    )

    assert _init_container_by_name(pod, "fetch-source") is None

    volume_mounts = {
        (vm.name, vm.mount_path)
        for vm in pod.pod.spec.containers[0].volume_mounts or []
    }
    assert ("v3io", "/empty/source") in volume_mounts

    # v3io_to_vol returns a plain dict (not a V1Volume), unlike the other mount_* helpers.
    v3io_volume = next(
        v
        for v in pod.pod.spec.volumes or []
        if (v["name"] if isinstance(v, dict) else v.name) == "v3io"
    )
    flex_volume_options = v3io_volume["flexVolume"].options
    assert flex_volume_options["container"] == "bigdata"
    assert flex_volume_options["subPath"] == "/project"

    assert "ADD ./source/code /home/mlrun_code" in _decoded_dockerfile(pod)


def test_buildah_backend_local_abs_path_source_passthrough():
    # parity with Kaniko's own edge case: an absolute local path is assumed valid inside the
    # build image already (e.g. baked into a custom base image) - no fetch, no mount.
    pod = _make_buildah_backend_pod(source="/opt/baked-in-source")

    assert _init_container_by_name(pod, "fetch-source") is None
    assert "ADD /opt/baked-in-source /home/mlrun_code" in _decoded_dockerfile(pod)


def test_buildah_backend_relative_path_source_raises():
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match="relative source"):
        _make_buildah_backend_pod(source="relative/path")


def test_buildah_backend_unsupported_scheme_raises_before_pod_construction():
    # fails fast at BuildRequest-resolution time, before scheduling a pod that would only fail
    # once the fetch-source init container actually runs `mlrun load-source`.
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match="ftp"):
        _make_buildah_backend_pod(source="ftp://example.com/file")


def test_buildah_backend_no_source_has_no_fetch_container_or_mount():
    pod = _make_buildah_backend_pod(source="")
    assert _init_container_by_name(pod, "fetch-source") is None
    assert "ADD" not in _decoded_dockerfile(pod)


def test_buildah_backend_inline_code_with_source_ignores_source():
    # inline_code takes precedence over source (Kaniko's own /empty-context rule) - no fetch,
    # no mount, no ADD line for the ignored source.
    pod = _make_buildah_backend_pod(
        source="git://github.com/some-org/some-repo.git#main",
        inline_code="print('hi')",
    )
    assert _init_container_by_name(pod, "fetch-source") is None
    assert "ADD" not in _decoded_dockerfile(pod)
