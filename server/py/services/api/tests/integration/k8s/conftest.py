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

import pytest
import yaml

import framework.utils.singletons.k8s

if typing.TYPE_CHECKING:
    import testcontainers.k3s

from kubernetes import client as k8s_client

from .utils import create_or_replace_k8s_resource, ensure_k8s_resource_deleted


@pytest.fixture(scope="session")
def docker_network():
    from testcontainers.core.network import Network

    with Network() as network:
        yield network


@pytest.fixture(scope="session")
def registry_container(docker_network):
    from testcontainers.registry import DockerRegistryContainer

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
def in_cluster_registry_url() -> str:
    return "distribution-registry:5000"


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
        .with_kwargs(
            privileged=True,
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


def _k8s_helper_from_config(
    cfg_path: str,
) -> framework.utils.singletons.k8s.K8sHelper:
    return framework.utils.singletons.k8s.K8sHelper(
        kube_config_path=cfg_path,
        silent=False,
        log=False,
    )


@pytest.fixture
def invalid_ssl_ca_k8s_helper(
    bad_ca_kubeconfig_path: str,
) -> framework.utils.singletons.k8s.K8sHelper:
    return _k8s_helper_from_config(bad_ca_kubeconfig_path)


@pytest.fixture
def valid_k8s_helper(valid_kubeconfig_path) -> framework.utils.singletons.k8s.K8sHelper:
    return _k8s_helper_from_config(valid_kubeconfig_path)


@pytest.fixture
def distribution_registry_k8s_service(
    registry_container,
    docker_network,
    valid_k8s_helper: framework.utils.singletons.k8s.K8sHelper,
):
    namespace = "default"
    name = "distribution-registry"
    port = 5000

    # Reload the container to ensure we have the latest network settings
    wrapped = registry_container.get_wrapped_container()
    wrapped.reload()

    # Get the IP address of the registry container
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

    # Create the service and endpoints in the k8s cluster
    # The service is used to access the registry from the k8s cluster
    # The endpoints are used to route traffic to the registry.
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
        valid_k8s_helper, "service", name, service, namespace
    )
    create_or_replace_k8s_resource(
        valid_k8s_helper, "endpoints", name, endpoints, namespace
    )

    yield {"ip": ip, "port": port}

    ensure_k8s_resource_deleted(valid_k8s_helper, "endpoints", name, namespace)
    ensure_k8s_resource_deleted(valid_k8s_helper, "service", name, namespace)
