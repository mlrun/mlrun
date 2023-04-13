(monitoring)=

# Monitor and alert

```{note}
Monitoring is supported by Iguazio's streaming technology, and open-source integration with Kafka.
```

```{note}
This is currently a beta feature.
```

The MLRun's model monitoring service includes built-in model monitoring and reporting capability. With monitoring you get
out-of-the-box analysis of:

- **Model performance**: machine learning models train on data. It is important you know how well they perform in production.
  When you analyze the model performance, it is important you monitor not just the overall model performance, but also the
  feature-level performance. This gives you better insights for the reasons behind a particular result
- **Data drift**: the change in model input data that potentially leads to model performance degradation. There are various
  statistical metrics and drift metrics that you can use in order to identify data drift.
- **Concept drift**: applies to the target. Sometimes the statistical properties of the target variable, which the model is
  trying to predict, change over time in unforeseen ways.
- **Operational performance**: applies to the overall health of the system. This applies to data (e.g., whether all the
  expected data arrives to the model) as well as the model (e.g., response time, and throughput). 

You have the option to set up notifications on various channels once an issue is detection. For example, you can set-up notification
to your IT via email and slack when operational performance metrics pass a threshold. You can also set-up automated actions, for example,
call a CI/CD pipeline when data drift is detected and allow a data scientist to review the model with the revised data.

Refer to the [**model monitoring & drift detection tutorial**](../tutorial/05-model-monitoring.html) for an end-to-end example.

**In this section**

```{toctree}
:maxdepth: 1

model-monitoring-deployment
initial-setup-configuration
```
