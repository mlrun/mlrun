(server-metrics)=
# Server metrics

MLRun collects anonymized system-size statistics, for example, project counts, artifact counts, run activity, serving endpoints, etc., and exports them to Prometheus via OpenTelemetry. 

In addition, MLRun records the processing time of every REST API call as a histogram whenever telemetry is enabled (see [REST call metrics](#rest-call-metrics) below).

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
Beyond the system-size gauges above, MLRun records processing time, request/response body size, and (for list calls) the number of objects returned for every REST API call, as OpenTelemetry histograms, exported to Prometheus. These are emitted from every API-bearing replica (the API chief and workers, and the alerts service).

This feature is enabled by default whenever the master switch is on. No extra flag is needed:
```
MLRUN_TELEMETRY__ENABLED=true
```
To disable REST metrics independently while keeping other telemetry on:
```
MLRUN_TELEMETRY__REST_METRICS__ENABLED=false
```

`system_id`, `status_code`, `resource`, and `project` are common to every instrument below. `resource` is the object type the route operates on (for example `functions`, `runs`, `artifacts`); `project` is set for project-scoped routes and empty otherwise. Health-check (`/healthz`) requests are excluded.

`method` is the real HTTP method, except a collection-returning GET is reported as the synthetic `"LIST"` value instead of `"GET"` — so list calls are distinguishable without a separate label. It's omitted entirely (not just empty) wherever it wouldn't vary: absent from `mlrun_rest_response_num_items`, since that metric only ever records `method="LIST"` calls by construction — a label that never varies within a metric adds nothing to query it by.

The four per-call metrics are all histograms — including items-returned, deliberately: it's a per-call value like duration or size, so a histogram preserves the per-call distribution (e.g. p95 list size) on top of the sum/count a plain counter would give.

|Metric name |Kind |Meaning       |
|---------------------|------|--------------------------------------------------------------------------------|
|mlrun_rest_request_duration_milliseconds|Histogram|Processing time (in milliseconds) of each REST call, from receipt to the full response being sent (excludes any background-task processing after the response completes).|
|mlrun_rest_request_size_kibibytes|Histogram|Size of the REST request body, in kibibytes.|
|mlrun_rest_response_size_kibibytes|Histogram|Size of the REST response body, in kibibytes.|
|mlrun_rest_response_num_items|Histogram|Number of objects returned by list calls (`method="LIST"` only).|
|mlrun_rest_metrics_sample_rate_ratio|Gauge|Currently configured `sample_rate` (see [Sampling](#sampling) below) — only carries `system_id`, no other attributes.|

The size histograms carry the OTel unit `KiBy` (kibibytes, 2^10 bytes), and their metric name already ends in `_kibibytes` to agree with it. See the [OTel<->Prometheus metric-metadata docs](https://opentelemetry.io/docs/specs/otel/compatibility/prometheus_and_openmetrics/#metric-metadata).

### Sampling
Not every call needs to be recorded to keep the metrics useful, so routine calls can be sampled:
```
MLRUN_TELEMETRY__REST_METRICS__SAMPLE_RATE=0.1
```
`sample_rate` (default `1.0`, i.e. no sampling) is the probability that a routine call's metrics are recorded. Failed calls (status >= 300), slow calls (processing time > 10 seconds), and calls with a large response (> 100 KiB) are always recorded regardless of the rate — those thresholds are fixed in code, not configurable. When sampling is enabled, compensate by dividing any count-based query by `sample_rate` to estimate the true call volume — or by `mlrun_rest_metrics_sample_rate_ratio` directly, to avoid hard-coding the current config value into every query.

### Example output
```
mlrun_rest_request_duration_milliseconds_count{system_id="f3a2b1c4d5e6", method="LIST", status_code="200", resource="functions", project="name1"} 134
mlrun_rest_request_duration_milliseconds_count{system_id="f3a2b1c4d5e6", method="GET", status_code="404", resource="runs", project="name1"}        2
mlrun_rest_request_duration_milliseconds_bucket{system_id="f3a2b1c4d5e6", method="LIST", status_code="200", resource="functions", project="name1", le="5"} 96
mlrun_rest_response_num_items_count{system_id="f3a2b1c4d5e6", status_code="200", resource="functions", project="name1"} 76
mlrun_rest_response_num_items_sum{system_id="f3a2b1c4d5e6", status_code="200", resource="functions", project="name1"} 812
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
# Average objects returned per list call, by object type
sum by (resource) (rate(mlrun_rest_response_num_items_sum[5m])) / sum by (resource) (rate(mlrun_rest_response_num_items_count[5m]))
# 95th-percentile objects returned per list call, by object type
histogram_quantile(0.95, sum by (resource, le) (rate(mlrun_rest_response_num_items_bucket[5m])))
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

