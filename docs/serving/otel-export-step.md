(otel-export-step)=
# Configure export of model monitoring metrics to OpenTelemetry

Export drift and performance signals from a serving graph to an external system (Datadog, Grafana Cloud, a user-hosted OpenTelemetry
Collector, or any OTLP-compatible backend). When configured, the results and metrics returned by `do_tracking()` in every monitoring window
are exported via OTLP to your configured endpoint. Each result becomes a named gauge:
`mlrun.model_monitoring.result.<name>`. 


The OTel default collector is configured by `mlrun.mlconf.telemetry.otlp_endpoint` on the API server.
You can modify it when running {py:class}`mlrun.serving.OTelMetricsExporter`.

```
 flow = function.set_topology("flow", engine="async")
flow.to(name="my_app", class_name="MyApp").to(
    class_name="mlrun.serving.OTelMetricsExporter",
    # endpoint, insecure default from mlconf.telemetry
    headers_source="file",
)
```