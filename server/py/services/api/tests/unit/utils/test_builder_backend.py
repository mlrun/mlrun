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
# seam, and (ML-12885) the Buildah backend: config-flip selection, the transparent
# Kaniko fallback for inputs Buildah can't handle yet, and the rootless pod spec.

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
def test_resolve_builder_backend_buildah_falls_back_for_cloud_registry(
    monkeypatch, target
):
    # cloud registries need credential-helper auth not yet on Buildah (ML-12886) -> fall back to Kaniko
    monkeypatch.setattr(config.httpdb.builder, "builder_backend", "buildah")
    backend = services.api.utils.builder.resolve_builder_backend(
        _make_build_request(image_target=target, registry=None)
    )
    assert isinstance(backend, services.api.utils.builder.KanikoBackend)


def test_resolve_builder_backend_buildah_falls_back_for_source(monkeypatch):
    # a source needing acquisition is not on Buildah yet (ML-12887) -> fall back to Kaniko
    monkeypatch.setattr(config.httpdb.builder, "builder_backend", "buildah")
    backend = services.api.utils.builder.resolve_builder_backend(
        _make_build_request(source="git://github.com/some-org/some-repo.git#main")
    )
    assert isinstance(backend, services.api.utils.builder.KanikoBackend)


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


def test_buildah_backend_rejects_unacquirable_source():
    # the source guard in resolve_builder_backend should route this to Kaniko; if the adapter is
    # somehow reached with an unacquirable source it must fail fast, not silently drop the source
    request = _make_build_request(source="git://github.com/some-org/some-repo.git#main")
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match="ML-12887"):
        services.api.utils.builder.BuildahBackend().make_build_pod(request)


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
