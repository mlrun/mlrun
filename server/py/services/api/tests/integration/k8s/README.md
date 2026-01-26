# K8s Integration Tests

This directory contains integration tests for MLRun API features that require a real Kubernetes cluster. Tests run against an ephemeral **k3s cluster** using [testcontainers](https://testcontainers-python.readthedocs.io/).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Docker Network (session-scoped)                 │
│                                                                     │
│  ┌─────────────┐   ┌─────────────────────┐   ┌───────────────────┐  │
│  │   k3s       │   │ distribution-       │   │ auth-registry     │  │
│  │  (cluster)  │   │ registry:5000       │   │ :5000 (htpasswd)  │  │
│  └──────┬──────┘   └──────────┬──────────┘   └─────────┬─────────┘  │
│         │                     │                        │            │
│         │    K8s Service + Endpoints (manual wiring)   │            │
│         ├─────────────────────┴────────────────────────┘            │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Pods (build pods, verification pods) inside k3s              │   │
│  │ - Can resolve services via cluster DNS                       │   │
│  │ - Can pull/push images to exposed registries                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## How It Works

1. **Session-scoped k3s cluster**: A single k3s container is started per pytest session via `testcontainers.k3s.K3SContainer`. All tests share this cluster.

2. **Container-to-k3s networking**: External Docker containers (registries, HTTP servers) run on the same Docker network as k3s. We expose them inside the cluster by creating headless `Service` + `Endpoints` resources pointing to the container's IP.

3. **DNS propagation wait**: Before tests use a service, we create a temporary pod to verify DNS resolution works. This is critical because build tools (Kaniko/Buildah) validate registry connectivity before starting.

4. **Insecure registry config**: k3s is configured with `/etc/rancher/k3s/registries.yaml` to allow insecure HTTP registries (no TLS).

## Available Fixtures

### Cluster & K8s Helper

| Fixture | Scope | Description |
|---------|-------|-------------|
| `docker_network` | session | Shared Docker network for all containers |
| `k3s` | session | The k3s cluster container |
| `session_k8s_helper` | session | `K8sHelper` instance for session-scoped fixtures |
| `valid_k8s_helper` | function | Fresh `K8sHelper` per test |
| `valid_kubeconfig_path` | function | Path to kubeconfig file |
| `raw_kubeconfig` | function | Parsed kubeconfig dict |

## Utility Functions

Located in `utils.py`:

```python
# Wait for pod to reach a phase (succeeded, failed, running, etc.)
wait_for_pod_phase(k8s, name, namespace, desired_phases, timeout_seconds=300)

# Get pod logs (safe, returns error message on failure)
dump_pod_logs(k8s, name, namespace) -> str

# Create or replace a K8s resource (handles 409 Conflict)
create_or_replace_k8s_resource(k8s, resource_type, resource_name, resource, namespace)

# Delete resource if exists (ignores 404)
ensure_k8s_resource_deleted(k8s, resource_type, resource_name, namespace)
```

## Writing New Tests

### Basic Test Template

```python
import pytest
from kubernetes import client as k8s_client

import framework.utils.singletons.k8s
from services.api.tests.integration.k8s.utils import wait_for_pod_phase, dump_pod_logs


@pytest.mark.integration
def test_my_feature(
    k3s_registry_service: str,          # Use registry if building images
    valid_kubeconfig_path: str,         # For creating K8sHelper
    monkeypatch,
):
    # Create K8sHelper from kubeconfig
    k8s = framework.utils.singletons.k8s.K8sHelper(
        kube_config_path=valid_kubeconfig_path,
        silent=False,
        log=False,
    )
    
    # Monkeypatch the singleton getter so MLRun code uses our helper
    monkeypatch.setattr(
        framework.utils.singletons.k8s,
        "get_k8s_helper",
        lambda *args, **kwargs: k8s,
    )
    
    # Create and run a pod
    pod_manifest = k8s_client.V1Pod(
        metadata=k8s_client.V1ObjectMeta(
            generate_name="my-test-",
            namespace="default",
        ),
        spec=k8s_client.V1PodSpec(
            restart_policy="Never",
            containers=[
                k8s_client.V1Container(
                    name="main",
                    image="gcr.io/iguazio/alpine:3.20",
                    command=["/bin/sh", "-c"],
                    args=["echo hello && sleep 1"],
                )
            ],
        ),
    )
    pod_name, namespace = k8s.create_pod(pod_manifest)
    
    # Wait for completion
    phase = wait_for_pod_phase(k8s, pod_name, namespace, {"succeeded", "failed"})
    logs = dump_pod_logs(k8s, pod_name, namespace)
    
    assert phase == "succeeded", f"Pod failed: {logs}"
    assert "hello" in logs
```

## Onboarding a New Service

To expose a new Docker container as a K8s service inside the cluster:

### 1. Create the Container Fixture

```python
@pytest.fixture(scope="session")
def my_service_container(docker_network, tmp_path_factory):
    """Create a container for my-service."""
    container = (
        DockerContainer("my-service:latest")
        .with_network(docker_network)
        .with_network_aliases("my-service")  # DNS name inside Docker network
        .with_name("my-service")
        .with_exposed_ports(8080)
        # Add volumes, env vars as needed
        # .with_volume_mapping(str(config_path), "/etc/config", "ro")
        # .with_env("MY_VAR", "value")
    )
    with container:
        yield container
```

### 2. Expose as K8s Service

```python
@pytest.fixture(scope="session")
def k3s_my_service(
    my_service_container,
    docker_network,
    session_k8s_helper: framework.utils.singletons.k8s.K8sHelper,
):
    """Expose my-service inside the k3s cluster."""
    with _expose_testcontainer_as_k8s_service(
        my_service_container,
        name="my-service",           # K8s service name
        port=8080,                   # Service port
        k8s_helper=session_k8s_helper,
    ) as service_url:
        yield service_url  # Returns "http://my-service:8080"
```

### 4. DNS Propagation

The `_expose_testcontainer_as_k8s_service` context manager automatically waits for DNS propagation by creating a temporary pod that runs `nslookup`. This ensures pods can resolve the service before tests start.

## Key Implementation Details

### Exposing Containers to K8s

The `_expose_testcontainer_as_k8s_service` function:

1. Gets the container's IP address from Docker network settings
2. Creates a headless K8s `Service` (no selector)
3. Creates `Endpoints` pointing directly to the container IP
4. Waits for cluster DNS to propagate
5. Cleans up on context exit

```python
# Simplified flow
ip = container.get_wrapped_container().attrs["NetworkSettings"]["Networks"][...]["IPAddress"]

service = V1Service(
    metadata=V1ObjectMeta(name=name),
    spec=V1ServiceSpec(ports=[V1ServicePort(port=port, target_port=port)]),
)

endpoints = V1Endpoints(
    metadata=V1ObjectMeta(name=name),
    subsets=[V1EndpointSubset(
        addresses=[V1EndpointAddress(ip=ip)],
        ports=[CoreV1EndpointPort(port=port)],
    )],
)
```

### Docker Registry Secrets

For authenticated registries, create a `kubernetes.io/dockerconfigjson` secret:

```python
docker_config = {
    "auths": {
        "registry:5000": {
            "username": "user",
            "password": "pass",
            "auth": base64.b64encode(b"user:pass").decode(),
        }
    }
}

secret = V1Secret(
    metadata=V1ObjectMeta(name="my-secret"),
    type="kubernetes.io/dockerconfigjson",
    data={".dockerconfigjson": base64.b64encode(json.dumps(docker_config).encode()).decode()},
)
```

## Troubleshooting

### Pod stuck in Pending

- Check if image is pullable: `k8s.v1api.read_namespaced_pod(name, namespace)` and look at `status.container_statuses`
- Verify registry is accessible from cluster

### DNS resolution fails

- The `_wait_for_service_dns` creates a debug pod; check its logs
- Verify service and endpoints exist: `k8s.v1api.read_namespaced_service(name, namespace)`

### Build pod fails

- Use `dump_pod_logs()` to get builder logs
- Check registry connectivity: builder validates push permissions before building

### Container IP not found

- Ensure container is on the same Docker network as k3s
- Call `container.get_wrapped_container().reload()` before reading network settings
