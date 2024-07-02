# Alerts and events

Events are alerts are the vehicle for informing you about possible problem situations.

- Event &mdash; something that happens in the system. An event may be a raw event (for example, drift-detection job detects drift in a given model) or a calculated event based on other events (CPU for a job is over 70% for one hour)
- Alert &mdash; a situation where a combination of events happening needs to be notified to the user
   - Alert config - the configuration defining the situation where an alert needs to be issued
   - Active alert - an alert config that was triggered and is actively issuing notifications
- Notifications &mdash; A means of notifying a user or the system of an alert, such as email, but also by invoking another event. See {ref}`notifications`.

## Alerts

### General configuration
There are two variables that control the alerts behavior: (AlertConfig)

- `alerts.mode` Either “enabled” or “disabled” to enable/disable the feature.
- `alerts.max_allowed` By default, 10000. Above this limit, any new alerts return an error.
- `alerts.max_criteria_count` By default, 100. Maximum number of events 

These values can be modified by the [support team](mailto:support@iguazio.com).

### Alert templates
Alerts use templates. The system comes with pre-defined templates. The pre-defined alerts can be used with MLRun applications. 
If you use non-MLRun applications (for example with model monitoring), you must configure an application-specific alert. 
For the pre-defined alerts, all you have to do is assign the project to an alert (and optionally tweak the default values). The pre-defined alerts are:
- data_drift_detected
- data_drift_suspected
- concept_drift_detected
- concept_drift_suspected
- model_performance_detected
- model_performance_suspected
- model_serving_performance_detected
- model_serving_performance_suspected
- mm_app_anomaly_detected
- mm_app_anomaly_suspected
- failed

### SDK

The SDK supports these alert operations:

- {py:func}`~mlrun.projects.MlrunProject.store_alert_config` &mdash; Create/modify an alert.
- {py:func}`~mlrun.projects.MlrunProject.get_alert_config` &mdash;  Retrieve an alert.
- {py:func}`~mlrun.projects.MlrunProject.reset_alert_config` &mdash; Reset an alert.
- {py:func}`~mlrun.projects.MlrunProject.delete_alert_config` &mdash; Delete an alert
- {py:func}`~mlrun.projects.MlrunProject.get_alert_template` &mdash; Retrieve a specific alert template.
- {py:func}`~mlrun.projects.MlrunProject.list_alert_templates` &mdash; Retrieve the list of all alert templates.
- {py:func}`~mlrun.projects.MlrunProject.list_alerts_configs` &mdash; Retrieve the list of alerts of a project.

### Creating alerts
You can select an alert type for a specific model, for example "drift detection" for a given model. You must specify 
the frequency of alerts, and the criteria for alerts (how many times in what time window, etc.). 
You can also configure email/slack details to send notifications.

```
notifications = [
            {
                "kind": "slack",
                "name": "",
                "message": "Running a job has failed",
                "severity": "warning",
                "when": ["now"],
                "condition": "failed",
                "secret_params": {
                    "webhook": "https://hooks.slack.com/services/",
                },
            }
                ]
alert_data = mlrun.alerts.alert.AlertConfig(
            project=project_name,
            name="failure",
            summary="Running a job has failed",
            severity="low",
            entity={"kind": "job", "project": project_name, "id": "*"},
            trigger={"events": ["failed"]},
            criteria=None,
            notifications=notifications,
        )

project.store_alert_config(alert_data)
```

```
%%writefile fn.py

def handler():
    raise Exception("This function intentionally fails")
```

### Monitoring alerts

The Alerts page in the UI shows the list of configured alerts, and which are active. In this page, you can acknowledge alerts and reset them. 
(If configured, you also get email/slack notifications.) You can also delete and modify alerts in this page.

