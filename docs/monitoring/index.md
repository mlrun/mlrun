(monitoring)=

# Model monitoring and alerts

In v1.6.0. MLRun introduces a new paradigm of model monitoring. The legacy mode and the new mode are both supported. The legacy 
mode is based on batch 

The MLRun's model monitoring service includes built-in model monitoring and reporting capabilities. With monitoring you get
out-of-the-box analysis of:

- **Model performance**: machine learning models train on data. It is important you know how well they perform in production.
  When you analyze the model performance, it is important you monitor not just the overall model performance, but also the
  feature-level performance. This gives you better insights for the reasons behind a particular result.
- **Data drift**: the change in model input data that potentially leads to model performance degradation. There are various
  statistical metrics and drift metrics that you can use to identify data drift.
- **Concept drift**: the statistical properties of the target variable (what the model is trying to predict) change over time. 
In other words, the meaning of the input data that the model was trained on has significantly changed over time,  and no longer matches the input data used to train the model. For this new data, accuracy of the model predictions is low. Drift analysis statistics are computed once an hour. See more details in <a href="https://www.iguazio.com/glossary/concept-drift/" target="_blank">Concept Drift</a>.
- **Operational performance**: applies to the overall health of the system. This applies to data (e.g., whether all the
  expected data arrives to the model) as well as the model (e.g., response time and throughput). 

You can set up notifications on various channels once an issue is detected. For example, notification
to your IT via email and Slack when operational performance metrics pass a threshold. You can also set-up automated actions, for example,
call a CI/CD pipeline when data drift is detected and allow a data scientist to review the model with the revised data.

**In this section**

```{toctree}
:maxdepth: 1

model-monitoring
model-monitoring-deployment
monitoring-models
```
