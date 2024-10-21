(genai-live-ops)=
# Gen AI liveOps 	

The liveOps stage ensures that the models are always performing optimally and adapting to new data. 

## Model Monitoring

MLRun includes tools for monitoring the performance of deployed models in real-time. This helps in identifying issues like model performance, operational performance, and concept and data drift.</br>
On top of the out-of-the-box analyses, you can easily {ref}`create model-monitoring applications <mm-applications>` of your own, tailored to meet your needs.</br>
Based on the monitoring data, MLRun can trigger automated retraining of models to ensure they remain accurate and effective over time.</br>
See full details in {ref}`model-monitoring-overview`.

## Alerts

Alerts inform you about potential or actual problem situations. Alerts can evaluate the same metrics as model mointoring: model performance, operational performance, concept/data drift, and on metrics that you define. 
Alerts use Git, Slack, and webhook, notifications. See full details in {ref}`alerts` and {ref}`notifications`.

## Guardrails

Guardrails are measures, guidelines, and frameworks designed to ensure the safe, reliable, and ethical use of AI-generated content. Typical goals are: 
aligning LLM functionalities with various legal and regulatory standards to avoid regulatory non-compliance; ensuring outputs are unbiased and fair, avoiding perpetuation of stereotypes or discriminatory practices; 
preventing toxicity: filtering out and preventing the generation of harmful or offensive content; 
preventing hallucination: minimizing the risk of LLMs generating factually incorrect or misleading information.


**See**
- {ref}`model-monitoring-overview`
- {ref}`mm-applications`
- {ref}`alerts-notifications`
- {ref}`realtime-monitor-drift-tutor`
- [Build & Deploy Custom (fine-tuned) LLM Models and Applications](https://github.com/mlrun/demo-llm-tuning/tree/main)
- [Large Language Model Monitoring](https://github.com/mlrun/demo-monitoring-and-feedback-loop/blob/main/notebook.ipynb), which includes using an LLM as a judge