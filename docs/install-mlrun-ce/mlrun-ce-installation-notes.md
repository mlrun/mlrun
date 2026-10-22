# MLRun CE installation notes

This page lists additional steps or configuration options you may need to follow for non-default MLRun CE installations.

**In this section**

- [Advanced chart configuration](#advanced-chart-configuration)
- [Opt out of components](#opt-out-of-components)
- [Using NFS storage](#using-nfs-storage)
- [Configuring the online feature store](#configuring-the-online-feature-store)
- [Using Azure Blob Storage for MLRun artifacts](#using-azure-blob-storage-for-mlrun-artifacts)
- [Ingress Configuration](#ingress-configuration)
- [Installing Spark Operator on non-mlrun namespace](#installing-spark-operator-on-non-mlrun-namespace)
- [Configure OTel](#configure-otel)

## Advanced chart configuration

Configurable values are documented in the `values.yaml`, and the `values.yaml` of all sub charts. Override those [in the normal methods](https://helm.sh/docs/chart_template_guide/values_files/).

See also the [MLRun CE values file reference](https://github.com/mlrun/ce/blob/development/charts/mlrun-ce/values.yaml)

## Opt out of components

The chart installs many components. You may not need them all in your deployment depending on your use cases.
To opt out of some of the components, use the following helm values:

```bash
--set pipelines.enabled=false \
--set kube-prometheus-stack.enabled=false \
--set spark-operator.enabled=false \
```

## Using NFS storage

If you are using NFS storage in your Kubernetes cluster, add these flags to the chart deployment command:

```
  --set kube-prometheus-stack.grafana.securityContext.runAsUser=1000 
  --set kube-prometheus-stack.grafana.securityContext.runAsGroup=1000 
  --set kube-prometheus-stack.grafana.securityContext.fsGroup=1000 
  --set kube-prometheus-stack.grafana.securityContext.fsGroupChangePolicy=OnRootMismatch 
  --set kube-prometheus-stack.grafana.initChownData.enabled
```

## Configuring the online feature store

The MLRun Community Edition supports the online feature store. To enable it, you need to first deploy a Redis service that is accessible to your MLRun CE cluster.
To deploy a Redis service, refer to the [Redis documentation](https://redis.io/learn/howtos/quick-start).

When you have a Redis service deployed, you can configure MLRun CE to use it by adding the following helm value configuration to your helm install command:

```bash
--set mlrun.api.extraEnvKeyValue.MLRUN_REDIS__URL=<redis-address>
```

## Using Azure Blob Storage for MLRun artifacts

MLRun CE can store MLRun artifact data in **Azure Blob Storage** instead of the default **S3-compatible (SeaweedFS)** setup used for MLRun CE. For a full Azure AKS installation guide, see {ref}`Install MLRun CE on Azure<azure-install>`.

Example for a custom values file, See also the [MLRun CE values file](https://github.com/mlrun/ce/blob/836384d05957875a5afdcf13e3f3a5975e76c950/charts/mlrun-ce/values.yaml#L33).

```yaml
storage:
  mode: azure-blob
  azure:
    containerName: "<your-container>"
    connectionString: "<your-connection-string>"
```

Example using flags and a storage connection string in the `helm` installtions:

```bash
--set storage.mode=azure-blob \
--set storage.azure.containerName='<your-container>' \
--set-string storage.azure.connectionString='<your-connection-string>'
```
## Ingress Configuration
If you prefer ingress-based access instead of NodePort, optionally add `values-ingress-override.yaml` to the install command. This switches exposed services to ClusterIP and enables ingress for MLRun UI, MLRun API, Nuclio, Jupyter, SeaweedFS Admin, Grafana, and Prometheus.

1. Download the {download}`ingress override values file template <./values-ingress-override.yaml.template>`.
2. Generate the override file. Export your FQDN and ingress class, then run:

```bash
export SYSTEM_FQDN="<system-fqdn>"
export INGRESS_CLASS_NAME="traefik"

envsubst < values-ingress-override.yaml.template > values-ingress-override.yaml
```

3. Install (or upgrade) with values file:

```bash
helm install mlrun-ce mlrun-ce/mlrun-ce \
  --namespace mlrun \
  --wait \
  --timeout 2000s \
  -f values-ingress-override.yaml
```

4. Configure DNS records for your ingress hostnames, pointing each host to your ingress controller's external IP or load balancer:

   - `mlrun.${SYSTEM_FQDN}`
   - `mlrun-api.${SYSTEM_FQDN}`
   - `nuclio.${SYSTEM_FQDN}`
   - `jupyter.${SYSTEM_FQDN}`
   - `seaweedfs.${SYSTEM_FQDN}`
   - `grafana.${SYSTEM_FQDN}`
   - `prometheus.${SYSTEM_FQDN}`

With ingress enabled, your applications are available at:

- MLRun UI - `https://mlrun.${SYSTEM_FQDN}`
- MLRun API - `https://mlrun-api.${SYSTEM_FQDN}`
- Nuclio - `https://nuclio.${SYSTEM_FQDN}`
- Jupyter Notebook - `https://jupyter.${SYSTEM_FQDN}`
- SeaweedFS Admin - `https://seaweedfs.${SYSTEM_FQDN}`
- Grafana - `https://grafana.${SYSTEM_FQDN}`
- Prometheus - `https://prometheus.${SYSTEM_FQDN}`

## Installing Spark Operator on non-mlrun namespace

By default Spark Operator jobNamespaces is set to "mlrun" namespace. If you are installing Spark Operator on a different namespace you need to set the jobNamespaces value accordingly

```bash
--set spark-operator.jobNamespaces={your-namespace}
```

## Configure OTel

MLRun CE running on Kubernetes integrates the OpenTelemetry Operator to bring metrics and distributed tracing to your ML workloads, with zero code changes required for standard use. 

Benefits of OTel:
- Automatic Python metrics from Nuclio functions — CPU, memory, GC, thread counts, system I/O — with no changes to function code
- Custom metrics — use the standard Python OTel SDK inside your function; the collector endpoint is pre-configured
- Metrics visible in Prometheus/Grafana out of the box — no extra exporters or sidecars needed
- Opt-in per function — only the functions you choose are instrumented; the rest of the platform is unaffected

When enabled, a single OTel Collector runs per namespace. Instrumented pods push metrics over OTLP to the Collector, which forwards
metrics to the in-cluster Prometheus instance via its OTLP write endpoint. Metrics are immediately available for querying in
Prometheus and visualizing in the bundled Grafana dashboard.

```{admonition} Note
Traces currently only go to the `debug` exporter. You can configure your own trace backend (for example, Jaeger, Tempo).
```

OTel is disabled by default. 

### Installation

Enable OTel by adding four flags to your Helm install command:

```
helm --namespace mlrun install my-mlrun \
    --set global.registry.url=<your-registry> \
    --set global.registry.secretName=registry-credentials \
    --set opentelemetry-operator.enabled=true \
    --set opentelemetry.namespaceLabel.enabled=true \
    --set opentelemetry.collector.enabled=true \
    --set opentelemetry.instrumentation.enabled=true \
    --wait mlrun/mlrun-ce
```
### Upgrade

Enable OTel by adding four flags to your Helm upgrade command:
```
helm --namespace mlrun upgrade my-mlrun \
    --set opentelemetry-operator.enabled=true \
    --set opentelemetry.namespaceLabel.enabled=true \
    --set opentelemetry.collector.enabled=true \
    --set opentelemetry.instrumentation.enabled=true \
    mlrun/mlrun-ce
```
### Verify the resources were created
```
kubectl -n mlrun get opentelemetrycollectors
kubectl -n mlrun get instrumentations
kubectl -n mlrun get pods | grep opentelemetry
```

### What gets instrumented
Instrumentation is opt-in per Nuclio function. To enable OTel injection on a function, add the annotation when deploying:
```
fn.with_annotations({
    "instrumentation.opentelemetry.io/inject-python": "mlrun-otel-instrumentation"
})
```
Once annotated, the OTel Operator injects an init container that sets up automatic Python instrumentation: no changes to the function code are required.

### Metrics
All metrics flow into Prometheus and are queryable in Grafana.
#### Process metrics
- `process_runtime_cpython_cpu_time_seconds_total`
- `process_runtime_cpython_context_switches_total`
- `process_runtime_cpython_cpu_utilization_ratio`
- `process_runtime_cpython_gc_count_bytes_total`
- `process_runtime_cpython_memory_bytes`
- `process_runtime_cpython_thread_count`
#### System metrics
 - CPU time + utilization
- disk I/O + operations + time
- memory usage + utilization
- network I/O + packets + errors + connections
- swap usage + utilization
- thread count


#### Custom metrics example (Python OTel SDK)
Once auto-instrumentation is active, the global MeterProvider is already configured. You can emit your own business metrics without any extra setup.

```
# function_with_otel.py
from opentelemetry import metrics
_counter = None
def init_context(context):
    global _counter
    # Auto-instrumentation sets up the MeterProvider before init_context runs.
    # Just call get_meter() to reuse it — no manual setup needed.
    meter = metrics.get_meter("nuclio.metrics")
    _counter = meter.create_counter(
        name="nuclio_requests_total",
        description="Total requests handled by this function",
        unit="1",
    )
def handler(context, event):
    _counter.add(1, {"function": "my-function"})
    return "ok"
```

And deploy your custom metrics (OTel is already enabled):
```python
import mlrun

fn = mlrun.code_to_function(
    name="my-otel-function",
    kind="nuclio",
    filename="function_with_otel.py",
    handler="handler",
    image="mlrun/mlrun",
)
# Opt in to OTel auto-instrumentation

fn.with_annotations(
    {"instrumentation.opentelemetry.io/inject-python": "mlrun-otel-instrumentation"}
)
fn.deploy(project=project.name)
```
After deploying and invoking the function, `nuclio_requests_total` appear in Prometheus alongside the automatic system and process metrics.