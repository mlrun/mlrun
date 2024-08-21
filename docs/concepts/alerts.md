(alerts)=
# Alerts 

Alerts are a mechanism for informing you about possible problem situations. 

{ref}`notifications` are used to notify you or the system of an alert.

**In this section**
- [System configuration](#system-configuration)
- [SDK](#sdk)
- [Predefined events](#predefined-events-eventkind)
- [Creating an alert](#creating-an-alert)
- [Alert reset policy](#alert-reset-policy)
- [Alert templates](#alert-templates)
- [Creating an alert with a template](#creating-an-alert-with-a-template)


**See also**
```{toctree}
:maxdepth: 1

drift-detection-alert
```

## System configuration 
These variables control the basic alert behavior: 
- `alerts.mode` &mdash; Enables/disables the feature. Enabled by default.
- `alerts.max_allowed` &mdash; Maximum number of alerts allowed to be configured, by default 10000. Any new alerts above this limit return an error.
- `alerts.max_criteria_count` &mdash; Maximum number of events. By default, 100.

These values can be modified by the [support team](mailto:support@iguazio.com).

## SDK

The SDK supports these alert operations:

- {py:func}`~mlrun.projects.MlrunProject.store_alert_config` &mdash; Create/modify an alert.
- {py:func}`~mlrun.projects.MlrunProject.get_alert_config` &mdash;  Retrieve an alert.
- {py:func}`~mlrun.projects.MlrunProject.reset_alert_config` &mdash; Reset an alert.
- {py:func}`~mlrun.projects.MlrunProject.delete_alert_config` &mdash; Delete an alert.
- {py:func}`~mlrun.projects.MlrunProject.get_alert_template` &mdash; Retrieve a specific alert template.
- {py:func}`~mlrun.projects.MlrunProject.list_alert_templates` &mdash; Retrieve the list of all alert templates.
- {py:func}`~mlrun.projects.MlrunProject.list_alerts_configs` &mdash; Retrieve the list of alerts of a project.


## Predefined events (`EventKind`)
The predefined event types are:
- `data_drift_detected` &mdash; A detected change in model input data that potentially leads to model performance degradation. 
- `data_drift_suspected` &mdash; A suspected change in model input data that potentially leads to model performance degradation. 
- `concept_drift_detected` &mdash; A detected change, over time, of  statistical properties of the target variable (what the model is predicting). 
- `concept_drift_suspected` &mdash; A suspected change, over time, of  statistical properties of the target variable (what the model is predicting). 
- `model_performance_detected` &mdash; A detected change of the overall model performance and/or feature-level performance. 
- `model_performance_suspected` &mdash; A suspected change of the overall model performance and/or feature-level performance. 
- `model_serving_performance_detected` &mdash; A detected change in how much time the prediction takes (i.e. the latency, measured in time units).
- `model_serving_performance_suspected` &mdash; A suspected change in how much time the prediction takes (i.e. the latency, measured in time units).
- `mm_app_anomaly_detected` &mdash; An alert based on user-defined metrics/results.
- `mm_app_anomaly_suspected` &mdash; An alert based on user-defined metrics/results.
- `failed` &mdash; The job failed.

See {ref}`monitoring-overview` for more details on drift and performance.

## Creating an alert
You can select an event type for a specific model, for example `data_drift_suspected`, for a given model. You can optionally specify 
the frequency of events, and the criteria for events (how many times in what time window, etc.). If not specified, it uses the defaults. 
See all of the {py:class}`alert configuration parameters<mlrun.alerts.alert.AlertConfig>`. You can configure Git, Slack, and webhook notifications for the alert. 

This example illustrates a Slack notification for drift detection on a model endpoint:

```python
notification = mlrun.model.Notification(
    kind="slack",
    name="slack_notification",
    message="A drift was detected",
    severity="warning",
    when=["now"],
    condition="failed",
    secret_params={
        "webhook": "https://hooks.slack.com/",
    },
).to_dict()

endpoints = mlrun.get_run_db().list_model_endpoints(project=project_name)
endpoint_id = endpoints[0].metadata.uid
result_endpoint = get_result_instance_fqn(endpoint_id, "myappv2", "data_drift_test")
notifications = [alert_objects.AlertNotification(notification=notification)]
alert_name = "drift_alert"
alert_summary = "A drift was detected"
entity_kind = alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT
event_name = alert_objects.EventKind.DATA_DRIFT_DETECTED
alert_data = mlrun.alerts.alert.AlertConfig(
    project=project_name,
    name=alert_name,
    summary=alert_summary,
    severity=alert_objects.AlertSeverity.LOW,
    entities=alert_objects.EventEntities(
        kind=entity_kind, project=project_name, ids=[result_endpoint]
    ),
    trigger=alert_objects.AlertTrigger(events=[event_name]),
    criteria=None,
    notifications=notifications,
)

project.store_alert_config(alert_data)
```


This example illustrates a Slack notification for a job failure:
```python
notification = mlrun.model.Notification(
    kind="slack",
    name="slack_notification",
    message="Running a job has failed",
    severity="warning",
    when=["now"],
    condition="failed",
    secret_params={
        "webhook": "https://hooks.slack.com/",
    },
).to_dict()
notifications = [alert_objects.AlertNotification(notification=notification)]
alert_name = "failure_alert"
alert_summary = "Running a job has failed"
entity_kind = alert_objects.EventEntityKind.JOB
event_name = alert_objects.EventKind.FAILED
run_id = "run-id"
alert_data = mlrun.alerts.alert.AlertConfig(
    project=project_name,
    name=alert_name,
    summary=alert_summary,
    severity=alert_objects.AlertSeverity.HIGH,
    entities=alert_objects.EventEntities(
        kind=entity_kind, project=project_name, ids=[run_id]
    ),
    trigger=alert_objects.AlertTrigger(events=[event_name]),
    notifications=notifications,
)
project.store_alert_config(alert_data)
```

## Alert reset policy

The {py:class}`mlrun.common.schemas.alert.ResetPolicy` specifies when to clear the alert and change the alert's status from active to inactive. When an alert 
becomes inactive, its notifications cease. When it is re-activated, notifications are renewed.
The `ResetPolicy` options are:
- manual &mdash; for manual reset of the alert
- auto &mdash; if the criteria contains a time period such that the alert is reset once there are no more invocations in the relevant time window.



## Alert templates
Alert templates simplify the creation of alerts by providing a predefined set of configurations. The system comes with several 
predefined templates that can be used with MLRun applications. 
If you use non-MLRun applications (for example, with model monitoring), you must configure an application-specific alert. 
The templates are cross-project objects. When generating an alert, you must assign the project to it. 
See the {py:meth}`alert template parameters<mlrun.common.schemas.alert.AlertTemplate>`.

## Creating an alert with a template

The system has a few pre-defined templates: `JobFailed`, `DataDriftDetected`, `DataDriftSuspected`.
When using a pre-defined template, you only need to supply:
- name: str
- project: str
- entity: EventEntity
- NotificationKind: a list of at least one notification

`summary`, `severity`, `trigger`, and `reset policy`, are pre-configured in the template.  
You can customize one or more of these fields when creating an alert from a template.

See the {py:meth}`AlertTemplate parameters<mlrun.common.schemas.alert.AlertTemplate>`.

This example illustrates a Slack notification for a job failure alert, using the predefined system template "JobFailed":

```python
job_fail_template = project.get_alert_template("JobFailed")
alert_from_template = mlrun.alerts.alert.AlertConfig(
    project=project_name,
    name="failure",
    template=job_fail_template,
)
entities = alert_objects.EventEntities(
    kind=alert_objects.EventEntityKind.JOB,
    project=project_name,
    ids=[run_id],
)
alert_from_template.with_entities(entities=entities)
alert_from_template.with_notifications(notifications=notifications)
project.store_alert_config(alert_from_template)

notification = mlrun.model.Notification(
    kind="slack",
    name="slack_notification",
    secret_params={
        "webhook": "https://hooks.slack.com/",
    },
).to_dict()

notifications = [alert_objects.AlertNotification(notification=notification)]
```
