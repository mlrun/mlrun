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
import contextlib
import json
import logging
import os
import pathlib

import kubernetes.client as k8s_client
import pytest
import testcontainers.k3s
import yaml
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from testcontainers.registry import DockerRegistryContainer

import mlrun.utils.helpers

import framework.utils.singletons.k8s
from .utils import (
    create_or_replace_k8s_resource,
    ensure_k8s_resource_deleted,
    wait_for_pod_phase,
)


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
    Generate a `registries.yaml` for k3s/containerd to allow insecure local registries.
    """
    cfg = {
        "mirrors": {
            "distribution-registry:5000": {
                "endpoint": ["http://distribution-registry:5000"]
            },
            "auth-registry:5000": {"endpoint": ["http://auth-registry:5000"]},
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
    with _expose_testcontainer_as_k8s_service(
        registry_container, "distribution-registry", 5000, session_k8s_helper
    ) as registry_url:
        # Strip http:// prefix - docker registry uses host:port format
        yield registry_url.replace("http://", "")


@pytest.fixture(scope="session")
def authenticated_registry_container(docker_network, tmp_path_factory):
    """Create a Docker registry with basic auth enabled using bcrypt htpasswd."""
    username = "testuser"
    password = "testpass"

    # Generate bcrypt htpasswd using httpd container
    htpasswd_content = _generate_htpasswd_via_docker(username, password)

    # Create temp directory for auth files
    auth_dir = tmp_path_factory.mktemp("registry-auth")
    htpasswd_path = auth_dir / "htpasswd"
    htpasswd_path.write_text(htpasswd_content)

    registry = (
        DockerContainer("registry:2")
        .with_network(docker_network)
        .with_network_aliases("auth-registry")
        .with_name("auth-registry")
        .with_exposed_ports(5000)
        .with_volume_mapping(str(auth_dir), "/auth", "ro")
        .with_env("REGISTRY_AUTH", "htpasswd")
        .with_env("REGISTRY_AUTH_HTPASSWD_REALM", "Registry Realm")
        .with_env("REGISTRY_AUTH_HTPASSWD_PATH", "/auth/htpasswd")
    )
    with registry:
        yield registry, username, password


@pytest.fixture(scope="session")
def k3s_authenticated_registry_service(
    authenticated_registry_container,
    docker_network,
    session_k8s_helper: framework.utils.singletons.k8s.K8sHelper,
):
    """Create a K8s Service routing to the authenticated registry, plus a docker config secret."""
    container, username, password = authenticated_registry_container
    registry_name = "auth-registry"
    port = 5000

    with _expose_testcontainer_as_k8s_service(
        container, registry_name, port, session_k8s_helper
    ) as registry_url:
        registry_host = registry_url.replace("http://", "")

        # Create docker config secret for registry auth
        secret_name = "auth-registry-secret"
        _create_docker_registry_secret(
            session_k8s_helper,
            secret_name=secret_name,
            registry=registry_host,
            username=username,
            password=password,
            namespace="default",
        )

        yield registry_host, secret_name, username, password

        try:
            session_k8s_helper.v1api.delete_namespaced_secret(
                name=secret_name, namespace="default"
            )
        except k8s_client.ApiException:
            pass


def _generate_htpasswd_via_docker(username: str, password: str) -> str:
    """Generate bcrypt htpasswd entry using httpd container via testcontainers."""
    with DockerContainer("httpd:2").with_command(
        f"htpasswd -Bbn {username} {password}"
    ) as container:
        # Wait for the container to finish and get logs
        container.get_wrapped_container().wait()
        logs = container.get_logs()
        # logs is a tuple of (stdout, stderr)
        return logs[0].decode("utf-8")


def _create_docker_registry_secret(
    k8s_helper: framework.utils.singletons.k8s.K8sHelper,
    secret_name: str,
    registry: str,
    username: str,
    password: str,
    namespace: str = "default",
):
    """Create a Kubernetes docker-registry secret."""

    # Build the docker config JSON
    auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
    docker_config = {
        "auths": {
            registry: {
                "username": username,
                "password": password,
                "auth": auth_string,
            }
        }
    }
    docker_config_json = json.dumps(docker_config)

    secret = k8s_client.V1Secret(
        metadata=k8s_client.V1ObjectMeta(name=secret_name, namespace=namespace),
        type="kubernetes.io/dockerconfigjson",
        data={
            ".dockerconfigjson": base64.b64encode(docker_config_json.encode()).decode()
        },
    )

    create_or_replace_k8s_resource(k8s_helper, "secret", secret_name, secret, namespace)


@pytest.fixture(scope="session")
def http_context_server(
    docker_network,
    session_k8s_helper: framework.utils.singletons.k8s.K8sHelper,
):
    """Create an HTTP server in K8s that serves a simple build http context."""
    name = "http-context-server"
    port = 8080

    # Create a tarball with a simple file and serve it via HTTP
    cmd = (
        "mkdir -p /data && "
        "echo 'FROM gcr.io/iguazio/alpine:3.20' > /data/Dockerfile && "
        "echo 'RUN echo hello > /hello.txt' >> /data/Dockerfile && "
        "tar -czf /context.tar.gz -C /data . && "
        "httpd -f -p 8080 -h /"
    )
    with (
        DockerContainer("busybox")
        .with_command(f'sh -c "{cmd}"')
        .with_exposed_ports(8080)
        .with_network(docker_network)
    ) as container:
        with _expose_testcontainer_as_k8s_service(
            container, name, port, session_k8s_helper
        ) as server_url:
            yield server_url


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


@contextlib.contextmanager
def _expose_testcontainer_as_k8s_service(
    container: DockerContainer,
    name: str,
    port: int,
    k8s_helper: framework.utils.singletons.k8s.K8sHelper,
    namespace: str = "default",
):
    # Reload the container to ensure we have the latest network settings
    wrapped = container.get_wrapped_container()
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

    create_or_replace_k8s_resource(k8s_helper, "service", name, service, namespace)
    create_or_replace_k8s_resource(k8s_helper, "endpoints", name, endpoints, namespace)

    # Wait for DNS to propagate before allowing tests to run.
    # This is critical for Kaniko which validates push permissions (via DNS lookup)
    # before starting the build.
    _wait_for_service_dns(k8s_helper, name, namespace)

    yield f"http://{name}:{port}"

    ensure_k8s_resource_deleted(k8s_helper, "endpoints", name, namespace)
    ensure_k8s_resource_deleted(k8s_helper, "service", name, namespace)


def _k8s_helper_from_config(
    cfg_path: str,
) -> framework.utils.singletons.k8s.K8sHelper:
    return framework.utils.singletons.k8s.K8sHelper(
        kube_config_path=cfg_path,
        silent=False,
        log=False,
    )
