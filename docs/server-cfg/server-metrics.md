(server-metrics)=
# Server metrics

MLRun collects anonymized system-size statistics, for example, project counts, artifact counts, run activity, serving endpoints, etc., and exports them to Prometheus via OpenTelemetry. 

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

