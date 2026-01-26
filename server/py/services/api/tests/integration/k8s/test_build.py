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

import uuid

import pytest
from kubernetes import client as k8s_client

import mlrun
import mlrun.common.schemas
import mlrun.runtimes.pod

import framework.utils.singletons.k8s
import services.api.utils.builder
from services.api.tests.integration.k8s.utils import dump_pod_logs, wait_for_pod_phase
from services.api.utils.image_builder.factory import ImageBuilderFactory


@pytest.mark.integration
@pytest.mark.parametrize("builder_kind", ["kaniko", "buildah"])
def test_function_build_and_run_image(
    k3s_registry_service: str,
    valid_kubeconfig_path: str,
    monkeypatch,
    builder_kind: str,
):
    k8s = framework.utils.singletons.k8s.K8sHelper(
        kube_config_path=valid_kubeconfig_path,
        silent=False,
        log=False,
    )
    _setup_builder_monkeypatches(monkeypatch, k8s, k3s_registry_service, builder_kind)

    project = "it"
    name = "build-run"
    tag = uuid.uuid4().hex[:12]
    image = f"{k3s_registry_service}/mlrun-it/{builder_kind}:{tag}"
    marker = f"mlrun-integration-{builder_kind}-{tag}"

    fn = mlrun.new_function(name=name, kind="job")
    fn.metadata.project = project
    fn.metadata.namespace = "default"
    fn.spec.build.image = image
    fn.spec.build.base_image = "gcr.io/iguazio/python:3.11-slim"
    fn.spec.build.commands = [f"echo {marker} > /mlrun_build_marker"]

    # Trigger build (non-interactive -> create build pod)
    built = services.api.utils.builder.build_runtime(
        auth_info=mlrun.common.schemas.AuthInfo(),
        runtime=fn,
        with_mlrun=False,
        interactive=False,
        force_build=True,
    )
    assert built is False
    assert fn.status.build_pod, "Expected build pod name to be recorded"

    _wait_for_build_pod(k8s, fn.status.build_pod)
    _verify_image_with_marker(k8s, image, marker, builder_kind)


@pytest.mark.integration
@pytest.mark.parametrize("builder_kind", ["kaniko", "buildah"])
def test_build_with_http_remote_context(
    k3s_registry_service: str,
    valid_kubeconfig_path: str,
    http_context_server: str,
    monkeypatch,
    builder_kind: str,
):
    """Test building an image with HTTP context source."""
    k8s = framework.utils.singletons.k8s.K8sHelper(
        kube_config_path=valid_kubeconfig_path,
        silent=False,
        log=False,
    )
    _setup_builder_monkeypatches(monkeypatch, k8s, k3s_registry_service, builder_kind)

    project = "it"
    tag = uuid.uuid4().hex[:12]
    image = f"{k3s_registry_service}/mlrun-it/{builder_kind}-http:{tag}"
    marker = f"http-context-{builder_kind}-{tag}"

    # Use the HTTP context server URL
    context_url = f"{http_context_server}/context.tar.gz"

    builder = ImageBuilderFactory.create_builder()
    runtime_spec = mlrun.runtimes.pod.KubeResourceSpec()

    kpod = builder.make_build_pod(
        project=project,
        context=context_url,
        dest=image,
        dockertext=f"FROM gcr.io/iguazio/alpine:3.20\nRUN echo {marker} > /mlrun_build_marker\n",
        name=f"build-http-{builder_kind}-{tag[:8]}",
        runtime_spec=runtime_spec,
    )

    # Set namespace on BasePod before accessing .pod (it regenerates on each access)
    kpod.namespace = "default"
    build_pod_name, build_namespace = k8s.create_pod(kpod.pod)
    _wait_for_build_pod(k8s, build_pod_name, build_namespace)
    _verify_image_with_marker(k8s, image, marker, builder_kind)


@pytest.mark.integration
@pytest.mark.parametrize("builder_kind", ["kaniko", "buildah"])
def test_build_with_git_context(
    k3s_registry_service: str,
    valid_kubeconfig_path: str,
    monkeypatch,
    builder_kind: str,
):
    """Test building an image with Git context source."""
    k8s = framework.utils.singletons.k8s.K8sHelper(
        kube_config_path=valid_kubeconfig_path,
        silent=False,
        log=False,
    )
    _setup_builder_monkeypatches(monkeypatch, k8s, k3s_registry_service, builder_kind)

    project = "it"
    tag = uuid.uuid4().hex[:12]
    image = f"{k3s_registry_service}/mlrun-it/{builder_kind}-git:{tag}"
    marker = f"git-context-{builder_kind}-{tag}"

    # Use a public Git repository
    # This uses GitHub's hello-world repo which has a minimal structure
    git_context = "git://github.com/docker-library/hello-world.git"

    builder = ImageBuilderFactory.create_builder()
    runtime_spec = mlrun.runtimes.pod.KubeResourceSpec()

    # Build with inline dockertext since the repo may not have a Dockerfile at root
    kpod = builder.make_build_pod(
        project=project,
        context=git_context,
        dest=image,
        dockertext=f"FROM gcr.io/iguazio/alpine:3.20\nRUN echo {marker} > /mlrun_build_marker\n",
        name=f"build-git-{builder_kind}-{tag[:8]}",
        runtime_spec=runtime_spec,
    )

    # Set namespace on BasePod before accessing .pod (it regenerates on each access)
    kpod.namespace = "default"
    build_pod_name, build_namespace = k8s.create_pod(kpod.pod)
    _wait_for_build_pod(k8s, build_pod_name, build_namespace)
    _verify_image_with_marker(k8s, image, marker, builder_kind)


def _setup_builder_monkeypatches(
    monkeypatch,
    k8s: framework.utils.singletons.k8s.K8sHelper,
    k3s_registry_service: str,
    builder_kind: str,
):
    """Common monkeypatch setup for builder tests."""
    monkeypatch.setattr(
        framework.utils.singletons.k8s,
        "get_k8s_helper",
        lambda *args, **kwargs: k8s,
    )
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.builder, "docker_registry", k3s_registry_service
    )
    monkeypatch.setattr(mlrun.mlconf.httpdb.builder, "docker_registry_secret", "")
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.builder, "insecure_pull_registry_mode", "enabled"
    )
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.builder, "insecure_push_registry_mode", "enabled"
    )
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.builder, "container_builder_kind", builder_kind
    )


def _wait_for_build_pod(
    k8s: framework.utils.singletons.k8s.K8sHelper,
    build_pod_name: str,
    namespace: str = "default",
    timeout_seconds: int = 600,
):
    """Wait for build pod to complete and return phase."""
    phase = wait_for_pod_phase(
        k8s=k8s,
        name=build_pod_name,
        namespace=namespace,
        desired_phases={"succeeded", "failed"},
        timeout_seconds=timeout_seconds,
    )
    if phase != "succeeded":
        logs = dump_pod_logs(k8s, build_pod_name, namespace)
        raise AssertionError(f"Build pod failed. Logs:\n{logs}")
    return phase


def _verify_image_with_marker(
    k8s: framework.utils.singletons.k8s.K8sHelper,
    image: str,
    marker: str,
    builder_kind: str,
    marker_path: str = "/mlrun_build_marker",
    timeout_seconds: int = 300,
):
    """Run a pod with the built image and verify the marker exists."""
    run_pod_manifest = k8s_client.V1Pod(
        metadata=k8s_client.V1ObjectMeta(
            generate_name=f"mlrun-it-{builder_kind}-",
            namespace="default",
        ),
        spec=k8s_client.V1PodSpec(
            restart_policy="Never",
            containers=[
                k8s_client.V1Container(
                    name="run",
                    image=image,
                    command=["/bin/sh", "-c"],
                    args=[f"cat {marker_path}"],
                )
            ],
        ),
    )
    run_pod_name, run_namespace = k8s.create_pod(run_pod_manifest)
    run_phase = wait_for_pod_phase(
        k8s=k8s,
        name=run_pod_name,
        namespace=run_namespace,
        desired_phases={"succeeded", "failed"},
        timeout_seconds=timeout_seconds,
    )
    logs = dump_pod_logs(k8s, run_pod_name, run_namespace)
    assert run_phase == "succeeded", f"Run pod failed. Logs:\n{logs}"
    assert marker in logs
