# MLRun CE development notes

This page contains notes for configuring your development system (after installation).

**In this section**

- [Change the deployment and jobs default PVC](#change-the-deployment-and-jobs-default-pvc)
- [Configure the user Jupyter conda environment](#configure-the-user-jupyter-conda-environment)
- [Configure TimescaleDB and Kafka for model monitoring](#configure-timescaledb-and-kafka-for-model-monitoring)
- [Configure OTel](#configure-otel)

## Change the deployment and jobs default PVC
A default PVC is created during the MLRun installation. If you modified the env vars before importing MLRun (to change the PVC), those values are overwritten. Change the PVC, after importing MLRun, by running this code:

```
import mlrun
mlrun.mlconf.storage.auto_mount_type = "pvc"
pvc_params = {
    "pvc_name": <your-pvc-name>,
    "volume_name": <volume-name>,
    "volume_mount_path": <container mount path>,
}
mlrun.mlconf.storage.auto_mount_params = ",".join(
    [f"{key}={value}" for key, value in pvc_params.items()]
)
```

## Configure the user Jupyter conda environment

The default Jupyter comes with a conda env named `mlrun`. This conda is not persistent.
If you install any packages on this conda env, and then the Jupyter pod gets restarted or deleted, those packages will be deleted.

To create a new, persistent, environment, run this in your Jupyter terminal, where `myenv` is the name of your environment:

```bash
# Create the virtual environment
conda create -n <myenv> python==3.11 -y

# Activate the virtual environment
conda activate <myenv>

# Make sure that ipykernel is installed
pip install --user ipykernel

# Add the new virtual environment to Jupyter
python -m ipykernel install --user --name <myenv> --display-name "Python (<myenv>)"
```

## Configure TimescaleDB and Kafka for model monitoring
TimescaleDB and Kafka are part of the default CE installations for model monitoring.

  [TimescaleDB](https://docs.timescale.com/self-hosted/latest/install/) is a PostgreSQL-based time-series database used as the TSDB backend for model monitoring. 
  Default connection values for CE:
  - `Host`: timescaledb.<namespace>.svc.cluster.local
  - `Port`: 5432
  - `Database`: postgres
  - `User`: postgres
  - `Password`: postgres
  
  [Kafka](https://github.com/bitnami/charts/tree/main/bitnami/kafka) is the streaming platform used for data flow between model monitoring components.
  Default connection values for CE:
  - `Brokers`: kafka-stream.<namespace>.svc.cluster.local:9092
  
  ### Configure data store profiles
  The connections are managed by using [data store profiles](../store/datastore.md#datastore-profiles). Data store profiles manage the connection credentials securely.
  ```python
  from mlrun.datastore.datastore_profile import (
      DatastoreProfileKafkaStream,
      DatastoreProfilePostgreSQL,
  )
  # Create and register TSDB profile
  tsdb_profile = DatastoreProfilePostgreSQL(
      name=tsdb_profile_name,
      user="postgres",
      password="postgres",
      host="timescaledb",
      port=5432,
      database="postgres",
  )
  project.register_datastore_profile(tsdb_profile)
  # Create and register stream profile
  stream_profile = DatastoreProfileKafkaStream(
      name=stream_profile_name,
      brokers="kafka-stream:9092",
      topics=[],
  )
  project.register_datastore_profile(stream_profile)
  # Set model monitoring credentials and enable the infrastructure
  project.set_model_monitoring_credentials(
      tsdb_profile_name=tsdb_profile.name,
      stream_profile_name=stream_profile.name,
  )
```
See more details, including additional configuration options, in {py:class}`~mlrun.projects.MlrunProject.set_model_monitoring_credentials`.

## Configure OTel

MLRun CE integrates the OpenTelemetry Operator to bring metrics and distributed tracing to your ML workloads, with zero code changes required for standard use. 
Benefits
- Automatic Python metrics from Nuclio functions — CPU, memory, GC, thread counts, system I/O — with no changes to function code
- Custom metrics and distributed traces — use the standard Python OTel SDK inside your function; the collector endpoint is pre-configured
- Metrics visible in Prometheus/Grafana out of the box — no extra exporters or sidecars needed
- Opt-in per function — only the functions you choose are instrumented; the rest of the platform is unaffected

When enabled, a single OTel Collector runs per namespace. Instrumented pods push metrics and traces over OTLP Metrics to the in-cluster Prometheus instance (prometheus-operated:9090/api/v1/otlp). They are immediately available for querying in Prometheus and visualizing in the bundled Grafana dashboard.

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

And deploy your custome metrics (OTel is already enabled):
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

fn.with_annotations({
    "instrumentation.opentelemetry.io/inject-python": "mlrun-otel-instrumentation"
})
fn.deploy(project=project.name)
```
After deploying and invoking the function, nuclio_requests_total will appear in Prometheus alongside the automatic system and process metrics.