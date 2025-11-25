(genai-live-ops)=
# Gen AI liveOps 	

The liveOps stage ensures that the models are always performing optimally and adapting to new data. 

## Model Monitoring

MLRun includes tools for monitoring the performance of deployed models in real-time. This helps in identifying issues like model performance, operational performance, and concept and data drift. The [MLRun hub](https://www.mlrun.org/hub/) has additional monitoring apps that you can [import to your project](../../runtimes/load-from-hub.md#modules). 

In addition to these apps, you can easily {ref}`create your own model monitoring applications<mm-applications>`, tailored to meet your needs. You can upload them to your own [custom hub](../../runtimes/load-from-hub.md#custom-hub), making them available to other users.

Based on the monitoring data, MLRun can trigger automated retraining of models to ensure they remain accurate and effective over time.</br>
See full details in {ref}`model-monitoring-overview`.

For examples of implementing model monitoring, see the {ref}`genai-02-mm-llm` tutorial, the [Banking LLM monitoring and feedback loop](https://github.com/mlrun/demo-monitoring-and-feedback-loop/blob/main/README.md) demo, and the [Banking Agent](https://github.com/mlrun/demo-banking-agent/blob/main/README.md) demo.

## Alerts

Alerts inform you about potential or actual problem situations. Alerts can evaluate the same metrics as model mointoring: model performance, operational performance, concept/data drift, and on metrics that you define. 
Alerts use Git, Slack, and webhook, notifications. See full details in {ref}`alerts` and {ref}`notifications`.

## Guardrails

Guardrails are measures, guidelines, and frameworks designed to ensure the safe, reliable, and ethical use of AI-generated content. Typical goals are: 
aligning LLM functionalities with various legal and regulatory standards to avoid regulatory non-compliance; ensuring outputs are unbiased and fair; avoiding perpetuation of stereotypes or discriminatory practices; 
preventing toxicity: filtering out and preventing the generation of harmful or offensive content; 
preventing hallucination: minimizing the risk of LLMs generating factually incorrect or misleading information.

Guardrails can be implemented in a [real-timeserving graph](../deployment/genai_serving_graph.md). The [Banking Agent](https://github.com/mlrun/demo-banking-agent/blob/main/README.md) demo has a few examples of guardrail implementation including using an LLM as a judge, a commonly used technique for implementing and evaluating guardrails.

**See also**
- {ref}`model-monitoring-overview`
- {ref}`genai-01-basic-tutorial`
- {ref}`realtime-monitor-drift-tutor`