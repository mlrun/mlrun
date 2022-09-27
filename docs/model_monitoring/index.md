(model-monitoring)=

# Model and data monitoring (beta)

```{note}
Model monitoring is based on Iguazio's streaming technology. Contact Iguazio to enable this feature.
```

MLRun's model monitoring service tracks the performance of models in production to help identify
potential issues with concept drift and prediction accuracy before they impact business goals.
Typically, model monitoring is used by devops for tracking model performance, and by data scientists to track model drift.
Two monitoring types are supported:
- Model operational performance (latency, requests per second, etc.).
- Drift detection &mdash; identifies potential issues with the model. See [Drift Analysis](./model-monitoring-deployment.html#drift-analysis) for more details.

Model monitoring provides warning alerts that can be sent to stakeholders for processing.

The model monitoring data can be viewed using Iguazio's user interface or through Grafana dashboards. Grafana is an interactive web 
application visualization tool that can be added as a service in the Iguazio platform. See [Model monitoring using Grafana dashboards](./model-monitoring-deployment.html#model-monitoring-using-grafana-dashboards) for more details.

**In this section**
  
```{toctree}
:maxdepth: 1

model-monitoring-deployment
initial-setup-configuration
```