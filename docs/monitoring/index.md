(monitoring-overview)=

# Model monitoring

In v1.6.0. MLRun introduces a {ref}`new paradigm of model monitoring <model-monitoring>`. 
The {ref}`legacy mode <legacy-model-monitoring>` is currently supported only for the CE version of MLRun.

The MLRun's model monitoring service includes built-in model monitoring and reporting capabilities. With monitoring you get
out-of-the-box analysis of:

- **Continuous Assessment**: Model monitoring involves the continuous assessment of deployed machine learning models in real-time. 
   It's a proactive approach to ensure that models remain accurate and reliable as they interact with live data.
- **Model performance**: machine learning models train on data. It is important you know how well they perform in production.
  When you analyze the model performance, it is important you monitor not just the overall model performance, but also the
  feature-level performance. This gives you better insights for the reasons behind a particular result.
- **Data drift**: the change in model input data that potentially leads to model performance degradation. There are various
  statistical metrics and drift metrics that you can use to identify data drift.
- **Concept drift**: the statistical properties of the target variable (what the model is predicting) change over time. 
In other words, the meaning of the input data that the model was trained on has significantly changed over time,  and no longer matches the input data used to train the model. For this new data, accuracy of the model predictions is low. Drift analysis statistics are computed once an hour. See more details in <a href="https://www.iguazio.com/glossary/concept-drift/" target="_blank">Concept Drift</a>.
- **Operational performance**: applies to the overall health of the system. This applies to data (e.g., whether all the
  expected data arrives to the model) as well as the model (e.g., response time and throughput). 

You can set up notifications on various channels once an issue is detected. For example, notification
to your IT via email and Slack when operational performance metrics pass a threshold. You can also set-up automated actions, for example,
call a CI/CD pipeline when data drift is detected and allow a data scientist to review the model with the revised data

## Common terminology
The following terms are used in all the model monitoring pages:
* **Total Variation Distance** (TVD) &mdash; The statistical difference between the actual predictions and the model's trained predictions.
* **Hellinger Distance** &mdash; A type of f-divergence that quantifies the similarity between the actual predictions, and the model's trained predictions.
* **Kullback–Leibler Divergence** (KLD) &mdash; The measure of how the probability distribution of actual predictions is different from the second model's trained reference probability distribution.
* **Model Endpoint** &mdash; A combination of a model and a runtime function that can be a deployed Nuclio function or a job runtime. One function can run multiple endpoints; however, statistics are saved per endpoint.

**In this section**

```{toctree}
:maxdepth: 1

model-monitoring
monitoring-models
model-monitoring-deployment
legacy-model-monitoring
```
