(server-metrics)=
# Server metrics

MLRun collects anonymized system-size statistics, for example, project counts, artifact counts, run activity, serving endpoints, etc., and exports them to Prometheus via OpenTelemetry. 

In addition, MLRun can record the processing time of every REST API call as a histogram (opt-in; see [REST call metrics](#rest-call-metrics) below).

## Metrics description
Every metric carries a system_id attribute (MLRun installation UUID). Project-scoped metrics additionally carry a project name. 

|Metric name |Attributes       |Meaning       |
|---------------------|--------------------------------|--------------------------------------------------------------------------------|
|mlrun_projects|system_id|Current number of projects in the installation|
|mlrun_functions||system_id, project, kind ∈ {job, serving, application, dask, mpijob, spark, nuclio, …}|Current number of functions of a given kind in a given project. Consolidates the original separate serving_functions / app_runtime_functions metrics via the kind attribute.
|mlrun_workflows|system_id, project|Current number of workflow definitions in the project  |
|mlrun_artifacts|system_id, project, kind ∈ {model, dataset, document, llm_prompt, other}|Current number of artifacts of a given kind in the project  |
|mlrun_runs|system_id, project, state ∈ {running, completed, failed, aborted}|Current number of runs in the project in each state (snapshot view)  |
|mlrun_pipeline_executions|system_id, project, state ∈ {running, completed, failed, aborted}|Current number of pipeline executions in the project in each state  |
|mlrun_alert_configurations|system_id, project|Current number of alert configurations in the project  |
|mlrun_alert_activations|system_id, project |Current number of active alert activations in the project|
|mlrun_model_endpoints|system_id, project, kind ∈ {realtime, batch}|Current number of registered model endpoints of a given kind. Consolidates the original separate realtime_endpoints / batch_endpoints metrics via the kind attribute.  |
|mlrun_model_monitoring_applications|system_id, project|Current number of model-monitoring applications in the project.  |

## Example output
```
mlrun_projects{system_id="f3a2b1c4d5e6f7a8"} 5
mlrun_artifacts{system_id="f3a2b1c4d5e6f7a8", project="name1", kind="model"}   8
mlrun_artifacts{system_id="f3a2b1c4d5e6f7a8", project="name2", kind="dataset"} 34
mlrun_artifacts{system_id="f3a2b1c4d5e6f7a8", project="name3", kind="other"}   1
mlrun_runs{system_id="f3a2b1c4d5e6f7a8", project="name4", state="completed"} 120
mlrun_runs{system_id="f3a2b1c4d5e6f7a8", project="name5", state="failed"}     3
```
## Example PromQL views
PromQL (Prometheus Query Language) is the language used to select and aggregate time series data in real time.
Typical output looks like:
```
# Total artifacts across the system right now
sum(mlrun_artifacts)
# Top 10 projects by artifact count
topk(10, sum by (project) (mlrun_artifacts))
# Project count trend (sample every hour over the last 7d)
mlrun_projects[7d:1h]
# Net artifact change over the last 24h
delta(sum(mlrun_artifacts)[24h:])
```

(rest-call-metrics)=
## REST call metrics
Beyond the system-size gauges above, MLRun can record the processing time of every REST API call as an OpenTelemetry histogram, exported to Prometheus. It is emitted from every API-bearing replica (the API chief and workers, and the alerts service).

This feature is opt-in and off by default. Enable it (in addition to the master `MLRUN_TELEMETRY__ENABLED`) with:
```
MLRUN_TELEMETRY__REST_METRICS__ENABLED=true
```

|Metric name |Attributes       |Meaning       |
|---------------------|--------------------------------|--------------------------------------------------------------------------------|
|mlrun_rest_request_duration_milliseconds|system_id, method, status_code, resource, project|Processing time (in milliseconds) of each REST call, exposed as a histogram (`_bucket` / `_sum` / `_count` series). `resource` is the object type the route operates on (for example `functions`, `runs`, `artifacts`); `project` is set for project-scoped routes and empty otherwise. Health-check (`/healthz`) requests are excluded.|

### Example output
```
mlrun_rest_request_duration_milliseconds_count{system_id="f3a2b1c4d5e6", method="GET", status_code="200", resource="functions", project="name1"} 134
mlrun_rest_request_duration_milliseconds_count{system_id="f3a2b1c4d5e6", method="GET", status_code="404", resource="runs", project="name1"}        2
mlrun_rest_request_duration_milliseconds_bucket{system_id="f3a2b1c4d5e6", method="GET", status_code="200", resource="functions", project="name1", le="5"} 96
```

### Example PromQL views
```
# Total REST calls recorded
sum(mlrun_rest_request_duration_milliseconds_count)
# Request rate (req/s) by object type
sum by (resource) (rate(mlrun_rest_request_duration_milliseconds_count[5m]))
# 95th-percentile latency (ms) across all calls
histogram_quantile(0.95, sum by (le) (rate(mlrun_rest_request_duration_milliseconds_bucket[5m])))
# Error rate (req/s) by status code
sum by (status_code) (rate(mlrun_rest_request_duration_milliseconds_count{status_code=~"4..|5.."}[5m]))
```

## Configure metrics
OpenTelemetry metrics are configured in `config.py`. Modify the configuration with a `configmap.yaml` that is applied on the mlrun service.

## Disable/enable OpenTelemetry 
Metrics are enabled by default. 
To disable the metrics collection:
```
MLRUN_TELEMETRY__ENABLED=false
```

To enable the metrics collection:
```
MLRUN_TELEMETRY__ENABLED=true
```

## Set the shared OTLP endpoint
The shared OTLP endpoint (gRPC or HTTP) is used by every OpenTelemetry feature. 
To set the endpoint:
```
MLRUN_TELEMETRY__OTLP_ENDPOINT=http://<server-name>:<port>
```

