(alerts)=
# Alerts 

The alert mechanism provides a flexible way to detect and respond to important system events, such as job failures or model drift. You can define alerts using conditions like “event X happens N times in T minutes,” and attach notifications that are sent when the alert is activated. 

**In this section**
- [System configuration](#system-configuration)
- [SDK](#sdk)
- [Predefined events](#predefined-events-eventkind)
- [Create an alert](#create-an-alert)
- [Create a model monitoring alert](#create-a-model-monitoring-alert)
- [Modify an alert](#modify-an-alert)
- [Alert reset policy](#alert-reset-policy)
- [Alert templates](#alert-templates)
- [Create an alert with a template](#create-an-alert-with-a-template)

**See also**
- {ref}`alert_activations`: When an alert is activated by its configured trigger, MLRun saves the activation records that you can list, filter, etc. 

## System configuration 
These variables control the basic alert behavior: 
- `alerts.mode` &mdash; Enables/disables the feature. Enabled by default.
- `alerts.max_allowed` &mdash; Maximum number of alerts allowed to be configured, by default 10000. Any new alerts above this limit return an error.
- `alerts.max_criteria_count` &mdash; Maximum number of events. By default, 100.
- `alerts.cooldown_reset_interval` &mdash; Interval at which alerts, whose `cooldown_period` have completed, are reset. By default, 15 seconds.

These values can be modified by the [support team](mailto:support@iguazio.com).

## SDK

The SDK supports these alert operations:
- {py:class}`~mlrun.alerts.alert.AlertConfig` &mdash; Create an alert
- {py:func}`~mlrun.projects.MlrunProject.store_alert_config` &mdash; Create/modify an alert
- {py:func}`~mlrun.projects.MlrunProject.get_alert_config` &mdash;  Retrieve an alert
- {py:func}`~mlrun.projects.MlrunProject.reset_alert_config` &mdash; Reset an alert
- {py:func}`~mlrun.projects.MlrunProject.delete_alert_config` &mdash; Delete an alert
- {py:func}`~mlrun.projects.MlrunProject.get_alert_template` &mdash; Retrieve a specific alert template
- {py:func}`~mlrun.projects.MlrunProject.list_alert_templates` &mdash; Retrieve the list of all alert templates
- {py:func}`~mlrun.projects.MlrunProject.list_alerts_configs` &mdash; Retrieve the list of alerts of a project

## Predefined events (`EventKind`)
The predefined event types ({py:class}`~mlrun.common.schemas.alert.EventKind`) are:
- `concept-drift-detected` &mdash; A detected change, over time, of  statistical properties of the target variable (what the model is predicting). 
- `concept-drift-suspected` &mdash; A suspected change, over time, of  statistical properties of the target variable (what the model is predicting). 
- `data-drift-detected` &mdash; An alert triggered by a detected change in model input data that potentially leads to model performance degradation. 
- `data-drift-suspected` &mdash; An alert triggered by a suspected change in model input data that potentially leads to model performance degradation. 
- `mm-app-anomaly-detected` &mdash; An alert triggered by a detected anomaly in user-defined metrics/results.
- `mm-app-anomaly-suspected` &mdash; An alert triggered by a suspected anomaly in user-defined metrics/results.
- `mm-app-failed` &mdash; A model monitoring app failed.
- `model-monitoring-lag-detected` &mdash; The monitoring writer falls behind the processing of inference events.
- `model-performance-detected` &mdash; A detected change of the overall model performance and/or feature-level performance. 
- `model-performance-suspected` &mdash; A suspected change of the overall model performance and/or feature-level performance. 
- `model-serving-performance-detected` &mdash; A detected change in how much time the prediction takes (i.e. the latency, measured in time units).
- `model-serving-performance-suspected` &mdash; A suspected change in how much time the prediction takes (i.e. the latency, measured in time units).
- `failed` &mdash; The job failed.

See {ref}`model-monitoring-overview` for more details on drift and performance.

## Create an alert
When creating an alert you can select an event type for a specific model, for example `data_drift_suspected` or any of the predefined events above.
You can optionally specify the frequency of the alert using the criteria field, which controls the threshold number of events in a given time window that triggers the alert.
If criteria is not specified, the default is `count=1` and `period=None`, in which case the alert triggers immediately upon the first matching event.

You can configure Slack, Git, and webhook {ref}`notifications` for alerts.

``` {Admonition} Note on run identification
Alerts track the job runs by name (`run.metadata.name`), not by the unique run UID. The run name can either be set explicitly or automatically generated when a job is executed. 
You can access the run name from the result of the `run_function` call, for example:
```python
run = project.run_function("my-function", handler="handler", local=True)
run_id = run.metadata.name
```
See all of the {py:class}`~mlrun.alerts.alert.AlertConfig` parameters. 



### Cooldown period

The `cooldown_period` parameter of `AlertConfig` can be used to delay resetting an alert. 
When the `cooldown_period` is active, the alert remains active for its duration, but incoming events are ignored:
**you do not receive continuous alerts for an ongoing situation**.
After the cooldown period expires and the alert is reset, it can be triggered again by new events.
The reset timing is approximate: reset happens when a periodic task runs, by default, every ~15s. 
By default, the `cooldown_period` is not set.

Guidelines:
- The `reset_policy` must be set to `auto`.
- The `cooldown_period` must be >0 and >=`cooldown_reset_interval`. See [System configuration](#system-configuration). (When `cooldown_period` is not set or set to 0, alerts are reset immediately.)
- Cooldown periods can be set as, for example, 1d, 3h, 5m, 15s, etc.

You can manually reset an alert at any time ({py:func}`~mlrun.projects.MlrunProject.reset_alert_config`), whether or not the `cooldown_period` is active.

### Example
This example illustrates creating an alert with a Slack notification for a job failure with defined criteria
including a [cooldown_period](#cooldown-period) of 30 seconds. 
This example uses `run_id`. You can set it to the run’s name (`run.metadata.name`), which is assigned when you run a job function.
The same run-name could be reused for multiple executions, especially in cases where functions are retried or triggered with a fixed name. In this example, the alert is triggered if 3 separate job runs with the same name fail within 10 minutes (even though each job run has a different internal UID).

```python
notification = mlrun.model.Notification(
    kind="slack",
    name="slack_notification",
    secret_params={
        "webhook": "https://hooks.slack.com/",
    },
).to_dict()

notifications = [alert_objects.AlertNotification(notification=notification)]
alert_name = "failure-alert"
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
    cooldown_period="30s",
    notifications=notifications,
)

# Save (and activate) the alert config:
project.store_alert_config(alert_data)
```
## Create a model monitoring alert

Model monitoring alerts notify you when measured input data and/or statistics/results produce unexpected results, the same as other alerts. The difference is that the configuration of a model monitoring alert is based on specific model endpoints and optionally result names, including wildcards. See the full parameter details in {py:func}`~mlrun.projects.MlrunProject.create_model_monitoring_alert_configs`. 
(You could also use `mlrun.alerts.alert.AlertConfig` to configure ModelEndpoint alerts, but `create_model_monitoring_alert_configs` is much easier to configure).

```{admonition} Important
Create model monitoring alerts after your serving function is deployed. When using a wildcard or when not specifying exact name of app+result (for example when not specifying results at all), the apps in question must already be running and generating some metrics, so that the `get_model_endpoint_monitoring_metrics` API call is able to extract the details for the specific ModelEndpoint.
```
This example illustrates creating a model monitoring alert to detect data drift, with a webhook notification for the alert.
```py
alert_configs = myproject.create_model_monitoring_alert_configs(
    # Name of the AlertConfig template
    name="alert-name",
    summary="user_template_summary_EventKind.DATA_DRIFT_DETECTED",
    # Retrieve metrics from these endpoints to configure the alert
    endpoints=myproject.list_model_endpoints(),
    # AlertTrigger event type
    events=[EventKind.DATA_DRIFT_DETECTED],
    notifications=[notifications],
    result_names=[],  # Can use wildcards
    severity=alert_constants.AlertSeverity.LOW,
    criteria=None,
    reset_policy=mlrun.common.schemas.alert.ResetPolicy.MANUAL,
)
for alert_config in alert_configs:
    myproject.store_alert_config(alert_config)
```
See also [model monitoring lag detection alerts](../model-monitoring/running-applications.md#lag-detection-alerts).
## Modify an alert

When you run `store_alert_config` on an existing alert:
- The alert is reset if you modify a field that affects the conditions that trigger the alert. These fields are:
   - Entity
   - Trigger
   - Criteria
 - The alert is not reset if you modify a field that affects the notifications that are sent or the result of the alert being activated. These fields are:
   - Description
   - Summary
   - Severity
   - Notifications
   
You can use the `force_reset` option when running `store_alert_config` to force a reset for fields that, by default, do not reset the alert.  By default, `force_reset` is set to false. 

## Alert reset policy

The {py:class}`~mlrun.common.schemas.alert.ResetPolicy` specifies when to clear the alert and change the alert's status from active to inactive. When an alert 
becomes inactive, its notifications cease. When it is re-activated, notifications are renewed.
The `ResetPolicy` options are:
- manual &mdash; for manual reset of the alert
- auto &mdash; the alert is reset immediately after it is triggered and its notifications are sent.

``` {Admonition} Note
If you change the `reset-policy` of an active alert from manual to auto, the alert is immediately reset. 
This ensures that the behavior aligns with the `auto-reset` behavior.
```


## Alert templates
Alert templates simplify the creation of alerts by providing a predefined set of configurations. The system comes with several 
predefined templates that can be used with MLRun applications. 
If you use non-MLRun applications (for example, with model monitoring), you must configure an application-specific alert. 
The templates are cross-project objects. When generating an alert, you must assign the project to it. 
See the {py:meth}`alert template parameters<mlrun.common.schemas.alert.AlertTemplate>`.

## Create an alert with a template

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

