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

import framework.utils.singletons.k8s
import services.api.utils.builder
from services.api.tests.integration.k8s.utils import dump_pod_logs, wait_for_pod_phase


@pytest.mark.integration
@pytest.mark.parametrize("builder_kind", ["kaniko", "buildah"])
def test_function_build_and_run_image(
    in_cluster_registry_url: str,
    distribution_registry_k8s_service,
    valid_kubeconfig_path: str,
    monkeypatch,
    builder_kind: str,
):
    k8s = framework.utils.singletons.k8s.K8sHelper(
        kube_config_path=valid_kubeconfig_path,
        silent=False,
        log=False,
    )
    monkeypatch.setattr(
        framework.utils.singletons.k8s,
        "get_k8s_helper",
        lambda *args, **kwargs: k8s,
    )

    # Configure builder to push to the local test registry
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.builder, "docker_registry", in_cluster_registry_url
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

    project = "it"
    name = "build-run"
    tag = uuid.uuid4().hex[:12]
    image = f"{in_cluster_registry_url}/mlrun-it/{builder_kind}:{tag}"
    marker = f"mlrun-integration-{builder_kind}-{tag}"

    fn = mlrun.new_function(name=name, kind="job")
    fn.metadata.project = project
    fn.metadata.namespace = "default"
    fn.spec.build.image = image
    fn.spec.build.base_image = "python:3.11-slim"
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

    build_pod_name = fn.status.build_pod
    build_namespace = "default"
    phase = wait_for_pod_phase(
        k8s=k8s,
        name=build_pod_name,
        namespace=build_namespace,
        desired_phases={"succeeded", "failed"},
        timeout_seconds=600,
    )
    if phase != "succeeded":
        logs = dump_pod_logs(k8s, build_pod_name, build_namespace)
        raise AssertionError(f"Build pod failed. Logs:\n{logs}")

    # Run a pod using the newly built image and validate marker exists
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
                    args=["cat /mlrun_build_marker"],
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
        timeout_seconds=300,
    )
    logs = dump_pod_logs(k8s, run_pod_name, run_namespace)
    assert run_phase == "succeeded", f"Run pod failed. Logs:\n{logs}"
    assert marker in logs
