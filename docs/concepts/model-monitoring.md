(model-monitoring-overview)=
# Model monitoring 

The MLRun's model monitoring service includes built-in model monitoring and reporting capabilities. With model monitoring you get
out-of-the-box analysis of:

- **Continuous Assessment**: Model monitoring involves the continuous assessment of deployed machine learning models in real-time. 
   It's a proactive approach to ensure that models remain accurate and reliable as they interact with live data.
- **Model performance**: Machine learning models train on data. It is important you know how well they perform in production.
  When you analyze the model performance, it is important you monitor not just the overall model performance, but also the
  feature-level performance. This gives you better insights for the reasons behind a particular result.
- **Data drift**: The change in model input data that potentially leads to model performance degradation. There are various
  statistical metrics and drift metrics that you can use to identify data drift.
- **Concept drift**: The statistical properties of the target variable (what the model is predicting) change over time. 
   In other words, the meaning of the input data that the model was trained on has significantly changed over time,  and no longer 
   matches the input data used to train the model. For this new data, accuracy of the model predictions is low. Drift analysis 
   statistics are computed once every 10 minutes (or more, configurable). See more details in <a href="https://www.iguazio.com/glossary/concept-drift/" target="_blank">Concept Drift</a>.
- **Operational performance**: The overall health of the system. This applies to data (e.g., whether all the
  expected data arrives to the model) as well as the model (e.g., response time and throughput). 
- **User-provided applications**: On top of the out-of-the-box analyses, you can easily create model-monitoring applications 
of your own to detect drifts and anomalies in a customized way.

You can set up alerts and notifications on various channels once an issue is detected. For example, notification
to your IT via email and Slack when operational performance metrics pass a threshold. You can also set-up automated actions, for example,
call a CI/CD pipeline when data drift is detected and allow a data scientist to review the model with the revised data.

```{admonition} Note
Model monitoring is currently in Tech Preview status. Migrating from 1.7.0 to later versions will not maintain backwards compatibility.
```

**In this section**

```{toctree}
:maxdepth: 1

../model-monitoring/index
../model-monitoring/model-monitoring-flow
../model-monitoring/monitoring-models
../model-monitoring/applications
../concepts/alerts
```
