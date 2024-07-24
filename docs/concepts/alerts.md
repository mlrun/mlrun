# Alerts 

Alerts are a mechanism for informing you about possible problem situations. The Alerts page in the UI shows the 
list of configured alerts, and indicates which are active. In this page, you can acknowledge, 
delete, and modify alerts and also reset them. 

Notifications are used to notify you or the system of an alert, such as through slack, git or webhook. See {ref}`notifications`.

## Configuration
These are the variables that control the basic alert behavior: 

- `alerts.mode` &mdash; Enables/disables the feature. Enabled by default.
- `alerts.max_allowed` &mdash; Maximum number of alerts allowed to be configured, by default: 10000. Any new alerts above this limit return an error.
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

## Alert templates
Alert templates simplify the creation of alerts by providing a predefined set of configurations. The system comes with several 
predefined templates that can be used with MLRun applications. 
If you use non-MLRun applications (for example, with model monitoring), you must configure an application-specific alert. 
The templates are cross-project objects. When generating an alert, the user must assign the project to it. The predefined alerts are:
- data_drift_detected &mdash; A detected change in model input data that potentially leads to model performance degradation. See {ref}`monitoring-overview`.
- data_drift_suspected &mdash; A suspected change in model input data that potentially leads to model performance degradation. See {ref}`monitoring-overview`.
- concept_drift_detected &mdash; A detected change, over time, of  statistical properties of the target variable (what the model is predicting). See {ref}`monitoring-overview`.
- concept_drift_suspected &mdash; A suspected change, over time, of  statistical properties of the target variable (what the model is predicting). See {ref}`monitoring-overview`.
- model_performance_detected &mdash; A detected change of the overall model performance and/or feature-level performance. See {ref}`monitoring-overview`.
- model_performance_suspected &mdash; A suspected change of the overall model performance and/or feature-level performance. See {ref}`monitoring-overview`.
- model_serving_performance_detected &mdash; 
- model_serving_performance_suspected &mdash; 
- mm_app_anomaly_detected &mdash; 
- mm_app_anomaly_suspected &mdash; 
- failed &mdash; The job failed.


## Creating an alert with a template

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
```
## Creating an alert without a template
You can select an alert type for a specific model, for example "drift detection" for a given model. You must specify 
the frequency of alerts, and the criteria for alerts (how many times in what time window, etc.). 
You can also configure email/Slack details to send notifications.

```python
import mlrun.common.schemas.alert as alert_objects
notification = mlrun.model.Notification(
kind="slack",
name="slack_notification",
message="Running a job has failed",
severity="warning",
when=["now"],
condition="failed",
secret_params={
"webhook": "https://hooks.slack.com/services/",
},
).to_dict()

notifications = [alert_objects.AlertNotification(notification=notification)]
entity_kind = alert_objects.EventEntityKind.JOB
event_name = alert_objects.EventKind.FAILED
run_id="test-func-handler"
alert_data = mlrun.alerts.alert.AlertConfig(
project=project_name,
name="failure",
summary="Running a job has failed",
severity=alert_objects.AlertSeverity.LOW,
entities=alert_objects.EventEntities(
kind=entity_kind, project=project_name, ids=[run_id]
),
trigger=alert_objects.AlertTrigger(events=[event_name]),
criteria=None,
notifications=notifications,
)
```