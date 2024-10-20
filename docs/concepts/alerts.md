(alerts)=
# Alerts 

Alerts are a mechanism for informing you about possible problem situations. 

```{admonition} Note
Alets are in Tech Preview state and disabled by default.
To enable, add an environment variable to the override-env configmap: `MLRUN_ALERTS__MODE: "enabled"`.
```

**In this section**
- [System configuration](#system-configuration)
- [SDK](#sdk)
- [Predefined events](#predefined-events-eventkind)
- [Creating an alert](#creating-an-alert)
- [Alert reset policy](#alert-reset-policy)
- [Alert templates](#alert-templates)
- [Creating an alert with a template](#creating-an-alert-with-a-template)

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
- `data-drift-detected` &mdash; A detected change in model input data that potentially leads to model performance degradation. 
- `data-drift-suspected` &mdash; A suspected change in model input data that potentially leads to model performance degradation. 
- `concept-drift-detected` &mdash; A detected change, over time, of  statistical properties of the target variable (what the model is predicting). 
- `concept-drift-suspected` &mdash; A suspected change, over time, of  statistical properties of the target variable (what the model is predicting). 
- `model-performance-detected` &mdash; A detected change of the overall model performance and/or feature-level performance. 
- `model-performance-suspected` &mdash; A suspected change of the overall model performance and/or feature-level performance. 
- `model-serving-performance-detected` &mdash; A detected change in how much time the prediction takes (i.e. the latency, measured in time units).
- `model-serving-performance-suspected` &mdash; A suspected change in how much time the prediction takes (i.e. the latency, measured in time units).
- `mm-app-anomaly-detected` &mdash; An alert based on user-defined metrics/results.
- `mm-app-anomaly-suspected` &mdash; An alert based on user-defined metrics/results.
- `failed` &mdash; The job failed.

See {ref}`model-monitoring-overview` for more details on drift and performance.

## Creating an alert
When creating an alert you can select an event type for a specific model, for example `data_drift_suspected` or any of the predefined events above.
You can optionally specify the frequency of the alert through the criteria field in the configuration (how many times in what time window, etc.). 
If not specified, it uses the default.
See all of the {py:class}`alert configuration parameters<mlrun.alerts.alert.AlertConfig>`. 
You can configure Git, Slack, and webhook notifications for the alert. 

When you run `store_alert_config`, the alert is automatically reset.

This example illustrates creating an alert with a Slack notification for drift detection on a model endpoint:

```python
# Define the slack notification object
notification = mlrun.model.Notification(
    kind="slack",
    name="slack_notification",
    secret_params={
        "webhook": "https://hooks.slack.com/",
    },
).to_dict()

endpoints = mlrun.get_run_db().list_model_endpoints(project=project_name)

endpoint_id = endpoints[0].metadata.uid

# Generate a unique ID for the EventEntity
result_endpoint = get_result_instance_fqn(endpoint_id, "myappv2", "data_drift_test")

# Construct a list of notifications to be included in the alert config
notifications = [alert_objects.AlertNotification(notification=notification)]

alert_name = "drift_alert"

# The summary you will see in the notification once it is invoked
alert_summary = "A drift was detected"

# Choose the MODEL_ENDPOINT_RESULT for the model monitoring alert
entity_kind = alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT

# The event that will trigger the alert
event_name = alert_objects.EventKind.DATA_DRIFT_DETECTED

# Create the alert data to be passed to the store_alert_config function
alert_data = mlrun.alerts.alert.AlertConfig(
    project=project_name,
    name=alert_name,
    summary=alert_summary,
    severity=alert_objects.AlertSeverity.LOW,
    entities=alert_objects.EventEntities(
        kind=entity_kind, project=project_name, ids=[result_endpoint]
    ),
    trigger=alert_objects.AlertTrigger(events=[event_name]),
    notifications=notifications,
)

# And finally store the alert config in the project
project.store_alert_config(alert_data)
```


This example illustrates creating an alert with a Slack notification for a job failure with defined criteria.
This alert gets triggered if the job fails 3 times in a 10 minute period.

```python
notification = mlrun.model.Notification(
    kind="slack",
    name="slack_notification",
    secret_params={
        "webhook": "https://hooks.slack.com/",
    },
).to_dict()

notifications = [alert_objects.AlertNotification(notification=notification)]
alert_name = "failure_alert"
alert_summary = "Running a job has failed"
entity_kind = alert_objects.EventEntityKind.JOB
event_name = alert_objects.EventKind.FAILED

# The job's run id that will be tracked
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
    criteria=alert_objects.AlertCriteria(period="10m", count=3),
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

This example illustrates a Slack notification for a job failure alert, using the predefined system template `JobFailed`:

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

notification = mlrun.model.Notification(
    kind="slack",
    name="slack_notification",
    secret_params={
        "webhook": "https://hooks.slack.com/",
    },
).to_dict()

notifications = [alert_objects.AlertNotification(notification=notification)]

alert_from_template.with_notifications(notifications=notifications)

project.store_alert_config(alert_from_template)
```
