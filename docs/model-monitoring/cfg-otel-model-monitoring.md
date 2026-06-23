(cfg-otel-model-monitoring)=
# Configure export of model monitoring metrics to OpenTelemetry

Export drift and performance signals from MLRun model monitoring to an external system (Datadog, Grafana Cloud, a user-hosted OpenTelemetry
Collector, or any OTLP-compatible backend). When configured, the results and metrics returned by `do_tracking()` in every monitoring window
are exported via OTLP to your configured endpoint. Each result becomes a named gauge:
`mlrun.model_monitoring.result.<name>`. The endpoint is configured for the the system. You can overwrite this setting per project.
(The OpenTelemetry option does not impact the default writing of results and metrics to TSDB.)

## SDK
- {py:class}`mlrun.serving.OTelMetricsExporter`: serving graph step that exports OTel metrics as a side-effect.
- {py:class}`mlrun.projects.MlrunProject.enable_model_monitoring`
- {py:meth}`mlrun.projects.MlrunProject.set_model_monitoring_function`





```
proj_obj.enable_model_monitoring(
    base_period=monitor_app_schedule, deploy_histogram_data_drift_app=True, wait_for_deployment=True,
    otlp_enabled=True)
```

### Configure the system-wide endpoint

### Override the system-wide endpoint
You can override the default per project. 

## Enable export per project
{py:meth}`mlrun.projects.MlrunProject.set_model_monitoring_function` `oltp_enabled`

## Add export to a serving graph step

 {py:class}`mlrun.serving.OTelMetricsExporter`