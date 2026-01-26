# Copyright 2025 Iguazio
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
import base64
import logging
import os
import pathlib
import typing

import kubernetes.client as k8s_client
import pytest
import yaml
from testcontainers.core.network import Network
from testcontainers.registry import DockerRegistryContainer

import mlrun.utils.helpers

import framework.utils.singletons.k8s
from .utils import wait_for_pod_phase

if typing.TYPE_CHECKING:
    import testcontainers.k3s


from .utils import create_or_replace_k8s_resource, ensure_k8s_resource_deleted


@pytest.fixture(scope="session")
def docker_network():
    with Network() as network:
        yield network


@pytest.fixture(scope="session")
def registry_container(docker_network):
    registry = (
        DockerRegistryContainer()
        .with_network(docker_network)
        .with_network_aliases("distribution-registry")
        .with_name("distribution-registry")
    )
    with registry:
        yield registry


@pytest.fixture(scope="session")
def k3s_registries_config_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    Generate a `registries.yaml` for k3s/containerd to allow an insecure local registry.
    """
    cfg = {
        "mirrors": {
            "distribution-registry:5000": {
                "endpoint": ["http://distribution-registry:5000"]
            }
        }
    }
    path = tmp_path_factory.mktemp("k3s") / "registries.yaml"
    yaml.safe_dump(cfg, path.open("w"))
    return str(path)


@pytest.fixture(scope="session")
def k3s(docker_network, registry_container, k3s_registries_config_path):
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("testcontainers").setLevel(logging.DEBUG)

    os.environ["TESTCONTAINERS_HOST_OVERRIDE"] = "host.docker.internal"
    import testcontainers.k3s

    container = (
        testcontainers.k3s.K3SContainer()
        .with_network(docker_network)
        .with_volume_mapping(
            k3s_registries_config_path, "/etc/rancher/k3s/registries.yaml", "ro"
        )
        .with_name("k3s")
    )
    with container:
        yield container


@pytest.fixture
def raw_kubeconfig(
    k3s: "testcontainers.k3s.K3SContainer",
) -> dict:
    return yaml.safe_load(k3s.config_yaml())


@pytest.fixture
def valid_kubeconfig_path(
    tmp_path: pathlib.Path,
    raw_kubeconfig: dict,
) -> str:
    path = tmp_path / "kubeconfig.yaml"
    yaml.safe_dump(raw_kubeconfig, path.open("w"))
    return str(path)


@pytest.fixture
def bad_ca_kubeconfig_path(
    tmp_path: pathlib.Path,
    raw_kubeconfig: dict,
) -> str:
    bad = raw_kubeconfig.copy()
    bad["clusters"][0]["cluster"]["certificate-authority-data"] = base64.b64encode(
        b"not-a-ca"
    )
    path = tmp_path / "kubeconfig-badca.yaml"
    yaml.safe_dump(bad, path.open("w"))
    return str(path)


@pytest.fixture
def invalid_ssl_ca_k8s_helper(
    bad_ca_kubeconfig_path: str,
) -> framework.utils.singletons.k8s.K8sHelper:
    return _k8s_helper_from_config(bad_ca_kubeconfig_path)


@pytest.fixture
def valid_k8s_helper(valid_kubeconfig_path) -> framework.utils.singletons.k8s.K8sHelper:
    return _k8s_helper_from_config(valid_kubeconfig_path)


@pytest.fixture(scope="session")
def session_k8s_helper(
    tmp_path_factory: pytest.TempPathFactory,
    k3s: "testcontainers.k3s.K3SContainer",
) -> framework.utils.singletons.k8s.K8sHelper:
    raw_kubeconfig = yaml.safe_load(k3s.config_yaml())
    path = tmp_path_factory.mktemp("kubeconfig") / "kubeconfig.yaml"
    yaml.safe_dump(raw_kubeconfig, path.open("w"))
    return _k8s_helper_from_config(str(path))


@pytest.fixture(scope="session")
def k3s_registry_service(
    registry_container,
    docker_network,
    session_k8s_helper: framework.utils.singletons.k8s.K8sHelper,
):
    """Create a K8s Service routing to the registry container, wait for DNS."""
    namespace = "default"
    name, port = "distribution-registry", 5000
    registry_url = f"{name}:{port}"

    # Reload the container to ensure we have the latest network settings
    wrapped = registry_container.get_wrapped_container()
    wrapped.reload()

    networks = wrapped.attrs["NetworkSettings"]["Networks"]
    ip = next(
        (
            details.get("IPAddress")
            for details in (networks or {}).values()
            if details.get("IPAddress")
        ),
        None,
    )
    if not ip:
        raise RuntimeError(
            f"Failed resolving registry container IP. Networks: {networks}"
        )

    service = k8s_client.V1Service(
        metadata=k8s_client.V1ObjectMeta(name=name, namespace=namespace),
        spec=k8s_client.V1ServiceSpec(
            ports=[k8s_client.V1ServicePort(port=port, target_port=port)],
        ),
    )
    endpoints = k8s_client.V1Endpoints(
        metadata=k8s_client.V1ObjectMeta(name=name, namespace=namespace),
        subsets=[
            k8s_client.V1EndpointSubset(
                addresses=[k8s_client.V1EndpointAddress(ip=ip)],
                ports=[k8s_client.CoreV1EndpointPort(port=port)],
            )
        ],
    )

    create_or_replace_k8s_resource(
        session_k8s_helper, "service", name, service, namespace
    )
    create_or_replace_k8s_resource(
        session_k8s_helper, "endpoints", name, endpoints, namespace
    )

    # Wait for DNS to propagate before allowing tests to run.
    # This is critical for Kaniko which validates push permissions (via DNS lookup)
    # before starting the build.
    _wait_for_service_dns(session_k8s_helper, name, namespace)

    yield registry_url

    ensure_k8s_resource_deleted(session_k8s_helper, "endpoints", name, namespace)
    ensure_k8s_resource_deleted(session_k8s_helper, "service", name, namespace)


def _k8s_helper_from_config(
    cfg_path: str,
) -> framework.utils.singletons.k8s.K8sHelper:
    return framework.utils.singletons.k8s.K8sHelper(
        kube_config_path=cfg_path,
        silent=False,
        log=False,
    )


@pytest.fixture(scope="session")
def http_context_server(
    session_k8s_helper: framework.utils.singletons.k8s.K8sHelper,
):
    """Create an HTTP server in K8s that serves a simple build context tarball.

    This fixture creates a ConfigMap with a simple Dockerfile and a pod running
    a busybox httpd server that serves it as a tarball.
    """
    namespace = "default"
    name = "http-context-server"
    port = 8080
    server_url = f"http://{name}:{port}"

    # Create a ConfigMap with a startup script that creates and serves the tarball
    # The script creates a minimal context directory with a Dockerfile
    startup_script = """#!/bin/sh
set -e
mkdir -p /www/context
echo 'FROM gcr.io/iguazio/alpine:3.18' > /www/context/Dockerfile
echo 'RUN echo hello > /hello.txt' >> /www/context/Dockerfile
cd /www/context && tar -czf /www/context.tar.gz .
cd /www && httpd -f -p 8080
"""

    configmap = k8s_client.V1ConfigMap(
        metadata=k8s_client.V1ObjectMeta(name=name, namespace=namespace),
        data={"startup.sh": startup_script},
    )

    pod = k8s_client.V1Pod(
        metadata=k8s_client.V1ObjectMeta(name=name, namespace=namespace),
        spec=k8s_client.V1PodSpec(
            restart_policy="Always",
            containers=[
                k8s_client.V1Container(
                    name="httpd",
                    image="gcr.io/iguazio/busybox:stable",
                    command=["/bin/sh", "/scripts/startup.sh"],
                    ports=[k8s_client.V1ContainerPort(container_port=port)],
                    volume_mounts=[
                        k8s_client.V1VolumeMount(name="scripts", mount_path="/scripts"),
                    ],
                )
            ],
            volumes=[
                k8s_client.V1Volume(
                    name="scripts",
                    config_map=k8s_client.V1ConfigMapVolumeSource(
                        name=name, default_mode=0o755
                    ),
                ),
            ],
        ),
    )

    service = k8s_client.V1Service(
        metadata=k8s_client.V1ObjectMeta(name=name, namespace=namespace),
        spec=k8s_client.V1ServiceSpec(
            selector={"app": name},
            ports=[k8s_client.V1ServicePort(port=port, target_port=port)],
        ),
    )

    # Add label to pod for service selector
    pod.metadata.labels = {"app": name}

    create_or_replace_k8s_resource(
        session_k8s_helper, "config_map", name, configmap, namespace
    )
    create_or_replace_k8s_resource(session_k8s_helper, "pod", name, pod, namespace)
    create_or_replace_k8s_resource(
        session_k8s_helper, "service", name, service, namespace
    )

    # Wait for pod to be running
    wait_for_pod_phase(
        k8s=session_k8s_helper,
        name=name,
        namespace=namespace,
        desired_phases={"running"},
        timeout_seconds=120,
    )

    # Wait for service DNS
    _wait_for_service_dns(session_k8s_helper, name, namespace)

    yield server_url

    # Cleanup
    ensure_k8s_resource_deleted(session_k8s_helper, "service", name, namespace)
    ensure_k8s_resource_deleted(session_k8s_helper, "pod", name, namespace)
    ensure_k8s_resource_deleted(session_k8s_helper, "config_map", name, namespace)


def _wait_for_service_dns(
    k8s_helper: framework.utils.singletons.k8s.K8sHelper,
    service_name: str,
    namespace: str,
    timeout_seconds: int = 5 * 60,
) -> None:
    """
    Wait until the Kubernetes service DNS is resolvable from within the cluster.

    This creates a temporary pod that attempts to resolve the service DNS name.
    Kaniko performs DNS lookups before building, so we must ensure DNS is propagated
    before running build tests.
    """

    # Create a pod that tries to resolve the DNS and exits
    pod = k8s_helper.v1api.create_namespaced_pod(
        namespace=namespace,
        body=k8s_client.V1Pod(
            metadata=k8s_client.V1ObjectMeta(
                generate_name=f"dns-check-{service_name}", namespace=namespace
            ),
            spec=k8s_client.V1PodSpec(
                restart_policy="OnFailure",
                containers=[
                    k8s_client.V1Container(
                        name="dns-check",
                        image="gcr.io/iguazio/busybox:stable",
                        command=[
                            "/bin/sh",
                            "-c",
                            f"nslookup {service_name}.{namespace}.svc.cluster.local",
                        ],
                    )
                ],
            ),
        ),
    )
    mlrun.utils.helpers.retry_until_successful(
        backoff=3,
        timeout=timeout_seconds,
        logger=logging,
        verbose=True,
        _function=wait_for_pod_phase,
        k8s=k8s_helper,
        name=pod.metadata.name,
        namespace=namespace,
        desired_phases={"succeeded"},
    )
