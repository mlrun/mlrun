(otel-export-step)=
# Export metrics to OpenTelemetry

Export custom metrics from a serving graph to an external system (Datadog, Grafana Cloud, a user-hosted OpenTelemetry
Collector, or any OTLP-compatible backend). When configured, the custom metricsß
are exported via OTLP to your configured endpoint. See the description of the exported results/metrics in [Export results and metrics via OTel](../model-monitoring/running-applications.md#export-results-and-metrics-via-otel).

This UI icon is used for OTel export steps: <img src="../_static/images/steps-custom.png" alt="steps-custom" width="30"/>.

The OTel default collector is configured by `mlrun.mlconf.telemetry.otlp_endpoint` on the API server.
You can modify it when running {py:class}`~mlrun.serving.OTelMetricsExporter`.

```
graph = function.set_topology("flow", engine="async")
graph.to(name="my_app", class_name="MyApp").to(
    class_name="mlrun.serving.OTelMetricsExporter",
    # endpoint, insecure default from mlconf.telemetry
    headers_source="file",
)
```