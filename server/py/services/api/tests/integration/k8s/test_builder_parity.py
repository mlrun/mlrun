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

# ML-12889: Buildah/Kaniko integration-tier parity - real pod builds against a k3s testcontainer,
# pushed to a plain local registry (see conftest.py). Covers what doesn't need cloud credentials:
# same-Dockerfile-same-image equivalence and the overlay/vfs storage drivers. Cloud-registry auth
# (ECR/ACR/GAR) parity needs real cloud credentials and is exercised instead in the system tier,
# against a real lab (see tests/system/runtimes/test_buildah_parity.py).
#
# NOTE: this module has not been run against real Docker in the session that wrote it (no Docker
# daemon available); the k3s<->registry Docker-network wiring in conftest.py is written to the
# standard testcontainers multi-container pattern but is unverified end-to-end. Run it first with
# real Docker access and expect to iterate on the networking before trusting it in CI.

import time

import pytest

import mlrun
import mlrun.common.schemas
from mlrun.config import config
from mlrun.runtimes import RuntimeKinds

import framework.utils.singletons.k8s
import services.api.utils.builder
from . import container_diff

_BUILD_TIMEOUT_SECONDS = 300
_POLL_INTERVAL_SECONDS = 5


@pytest.fixture
def cluster_k8s_helper(valid_kubeconfig_path, monkeypatch):
    """Point the builder's k8s singleton at the real k3s cluster instead of the mocked helper
    the unit tests use - build_image/build_runtime create and poll a real pod through this.
    """
    helper = framework.utils.singletons.k8s.K8sHelper(
        namespace="default",
        kube_config_path=valid_kubeconfig_path,
        silent=False,
        log=False,
    )
    monkeypatch.setattr(
        framework.utils.singletons.k8s,
        "get_k8s_helper",
        lambda *args, **kwargs: helper,
    )
    return helper


def _build_and_wait_for_terminal_phase(
    function, cluster_k8s_helper, timeout_seconds=_BUILD_TIMEOUT_SECONDS
) -> str:
    services.api.utils.builder.build_runtime(
        mlrun.common.schemas.AuthInfo(),
        function,
        with_mlrun=False,
        interactive=False,
        force_build=True,
    )
    pod_name = function.status.build_pod
    deadline = time.time() + timeout_seconds
    phase = "pending"
    while time.time() < deadline:
        phase = cluster_k8s_helper.get_pod_phase(pod_name)
        if phase in ("succeeded", "failed"):
            return phase
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Build pod {pod_name} did not reach a terminal phase: last seen {phase}"
    )


def _make_build_function(
    name, backend, registry_endpoint, monkeypatch, **build_overrides
):
    monkeypatch.setattr(config.httpdb.builder, "builder_backend", backend)
    monkeypatch.setattr(config.httpdb.builder, "docker_registry", registry_endpoint)
    monkeypatch.setattr(config.httpdb.builder, "insecure_push_registry_mode", "enabled")
    monkeypatch.setattr(config.httpdb.builder, "insecure_pull_registry_mode", "enabled")

    function = mlrun.new_function(
        name, "default", kind=RuntimeKinds.job, image="alpine:3.20"
    )
    function.spec.build.commands = build_overrides.get(
        "commands", ["echo mlrun-parity-test"]
    )
    function.spec.build.image = f".{name}"
    return function


@pytest.mark.integration
@pytest.mark.parametrize("storage_driver", ["overlay", "vfs"])
def test_buildah_kaniko_same_dockerfile_same_image(
    monkeypatch,
    cluster_k8s_helper,
    registry_endpoint,
    registry_host_endpoint,
    storage_driver,
):
    """Build the same commands-only function via both backends, push to the same local
    registry, and assert equivalence via container-diff - not byte-identical (timestamps and
    filesystem metadata always differ between engines), but no config/file/pip/apt diff.
    """
    if not container_diff.is_installed():
        pytest.skip(
            "container-diff is not installed - see CONTRIBUTING.md's Testing section"
        )

    monkeypatch.setattr(config.httpdb.builder, "buildah_storage_driver", storage_driver)

    pushed_images = {}
    for backend in ("kaniko", "buildah"):
        function = _make_build_function(
            f"parity-{backend}-{storage_driver}",
            backend,
            registry_endpoint,
            monkeypatch,
        )
        phase = _build_and_wait_for_terminal_phase(function, cluster_k8s_helper)
        assert phase == "succeeded", (
            f"{backend} build did not succeed: pod phase {phase}"
        )
        pushed_images[backend] = (
            f"{registry_host_endpoint}/{function.spec.build.image.lstrip('.')}"
        )

    container_diff.assert_images_equivalent(
        pushed_images["kaniko"], pushed_images["buildah"], insecure=True
    )


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["kaniko", "buildah"])
def test_failed_build_drives_function_to_error_state(
    monkeypatch, cluster_k8s_helper, registry_endpoint, backend
):
    """The failure-contract invariant end-to-end, for both backends: a build that fails inside
    the pod (here, a command that exits non-zero) must leave the pod Failed and, once the
    caller polls the build status (see functions.py's build-status endpoint), the function
    state resolved to `error` - not silently `ready` or stuck `deploying`.
    """
    function = _make_build_function(
        f"parity-fail-{backend}",
        backend,
        registry_endpoint,
        monkeypatch,
        commands=["exit 1"],
    )
    phase = _build_and_wait_for_terminal_phase(function, cluster_k8s_helper)
    assert phase == "failed"
    assert (
        mlrun.common.schemas.FunctionState.get_function_state_from_pod_state(phase)
        == mlrun.common.schemas.FunctionState.error
    )
