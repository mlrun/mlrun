(otel-export-step)=
# Export metrics to OpenTelemetry

You can export metrics from a serving graph to an external system (Datadog, Grafana Cloud, a user-hosted OpenTelemetry
Collector, or any OTLP-compatible backend). When configured, the metrics
are exported via OTLP to your configured endpoint. 

This UI icon is used for OTel export steps: <img src="../_static/images/steps-custom.png" alt="steps-custom" width="30"/>.

The default OTel collector is configured on the API server. See [Export results and metrics via OTel](../model-monitoring/running-applications.md#export-results-and-metrics-via-otel). 

Add a step for exporting metrics with {py:meth}`~mlrun.serving.OTelMetricsExporter`:

```
graph = function.set_topology("flow", engine="async")
graph.to(name="my_step", class_name="MyStep").to(
    class_name="mlrun.serving.OTelMetricsExporter",
    headers_source="file",
)
```