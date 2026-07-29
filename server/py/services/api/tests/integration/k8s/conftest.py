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
    import testcontainers.core.container
    import testcontainers.core.network
    import testcontainers.k3s


@pytest.fixture(scope="session")
def _test_network() -> "testcontainers.core.network.Network":
    # ML-12889: a shared Docker network so pods inside k3s can route to the sibling
    # registry container by IP (they get their own netns via flannel, so they can't reach
    # the registry through the host's loopback the way the test process itself can).
    import testcontainers.core.network

    network = testcontainers.core.network.Network()
    with network:
        yield network


@pytest.fixture(scope="session")
def k3s(_test_network: "testcontainers.core.network.Network"):
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("testcontainers").setLevel(logging.DEBUG)

    # Ensure that the testcontainers library can find the Docker socket when running inside a Docker container.
    # This env var is parsed by testcontainers at import time.
    os.environ["TESTCONTAINERS_HOST_OVERRIDE"] = "host.docker.internal"
    import testcontainers.k3s

    container = (
        testcontainers.k3s.K3SContainer()
        .with_kwargs(privileged=True)
        .with_network(_test_network)
    )
    with container:
        yield container


@pytest.fixture(scope="session")
def registry_container(
    _test_network: "testcontainers.core.network.Network",
    k3s: "testcontainers.k3s.K3SContainer",  # not used directly - ordering only, see docstring
) -> "testcontainers.core.container.DockerContainer":
    """A plain ``registry:2`` container the build pods push to, on the same Docker network
    as ``k3s`` so pods scheduled inside it can reach the registry by IP.

    ML-12889: shared by the Buildah/Kaniko parity integration tests - both backends push
    here so their output images can be compared with ``container-diff``. Whether a k3s pod's
    egress actually routes to a sibling container's bridge IP depends on the cluster's CNI
    backend (flannel's default vxlan/host-gw backends SNAT pod egress through the node's own
    interface, which is on this same bridge network) - this is the standard testcontainers
    multi-container topology, but hasn't been run against real Docker in this session; verify
    reachability first if pods can't reach ``registry_endpoint`` below.
    """
    import testcontainers.core.container

    container = testcontainers.core.container.DockerContainer("registry:2")
    container.with_exposed_ports(5000)
    container.with_network(_test_network)
    with container:
        yield container


@pytest.fixture
def registry_endpoint(
    registry_container: "testcontainers.core.container.DockerContainer",
) -> str:
    """The registry's address as reachable *from a pod scheduled inside the k3s cluster*:
    its IP on the shared Docker network (pods can't resolve Docker network aliases - they use
    the cluster's own CoreDNS, not the Docker embedded DNS)."""
    ip = registry_container.get_wrapped_container().attrs["NetworkSettings"][
        "Networks"
    ][_test_network_name(registry_container)]["IPAddress"]
    return f"{ip}:5000"


@pytest.fixture
def registry_host_endpoint(
    registry_container: "testcontainers.core.container.DockerContainer",
) -> str:
    """The registry's address as reachable *from the test process itself* (host-published
    port), for pulling/diffing the images the build pods pushed."""
    host = registry_container.get_container_host_ip()
    port = registry_container.get_exposed_port(5000)
    return f"{host}:{port}"


def _test_network_name(
    container: "testcontainers.core.container.DockerContainer",
) -> str:
    networks = container.get_wrapped_container().attrs["NetworkSettings"]["Networks"]
    # the container is only ever attached to the one shared _test_network fixture.
    return next(iter(networks))


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
def valid_k8s_helper(valid_kubeconfig_path):
    return _k8s_helper_from_config(valid_kubeconfig_path)
