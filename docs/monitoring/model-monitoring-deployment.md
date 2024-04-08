(model-monitoring)=
# Model monitoring user flow
```{note}
This is currently a beta feature. 
```
This page gives an overview of the model monitoring feature. See a complete example in the tutorial [Model monitoring and drift detection](../tutorials/05-model-monitoring.html).

## Architecture

When you enable model monitoring, you effectively deploy three components:
- application controller function: handles the monitoring processing and the triggers the apps that trigger the writer. The frequency is determined by `base_period`. 
- writer function: writes to the database and outputs alerts.
- stream function: monitors the log of the data stream. It is triggered when a new log entry is detected. The monitored data is used to create real-time dashboards, detect drift, and analyze performance.

## APIs

The model monitoring APIs are configured per project. The APIs are:

- {py:meth}`~mlrun.db.httpdb.HTTPRunDB.enable_model_monitoring` &mdash; brings up the controller and schedules it according to the `base_period`; deploys the writer.
- {py:meth}`~mlrun.projects.MlrunProject.disable_model_monitoring` &mdash; disables the controller. 
- {py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_function` &mdash; Update or set a monitoring function to the project. (Monitoring does not start until the function is deployed.) 
- {py:meth}`~mlrun.projects.MlrunProject.create_model_monitoring_function` &mdash; creates a function but does not set it. It's useful for troubleshooting, since it does  not register the function to the project.
- {py:meth}`~mlrun.projects.MlrunProject.list_model_monitoring_functions` &mdash; Retrieves a list of all the model monitoring functions.
- {py:meth}`~mlrun.projects.MlrunProject.remove_model_monitoring_function` &mdash; Removes the specified model-monitoring-app function from the project and from the DB.
- {py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_credentials` &mdash; Sets the Kafka or SQL credentials to be used by the project's model monitoring infrastructure functions. 

## Configuration flow

### Enable model monitoring

Enable model monitoring for a project with {py:meth}`~mlrun.projects.MlrunProject.enable_model_monitoring`.
The controller runs, by default, every 10 minutes, which is also the minimum interval. 
You can modify the frequency with the parameter `base_period`. 
To change the `base_period`, first run `disable_model_monitoring`, then run `enable_model_monitoring`  
with the new `base_period` value. 

### Log the model with training data
See the parameter descriptions in {py:meth}`~mlrun.projects.MlrunProject.log_model`.


### Import, enable monitoring, and deploy the serving function

Use the [v2_model_server serving](https://www.mlrun.org/hub/functions/master/v2-model-server/) function from the MLRun function hub.

Add the model to the serving function's routing spec ({py:meth}`~mlrun.runtimes.ServingRuntime.add_model`), 
enable monitoring on the serving function ({py:meth}`~mlrun.runtimes.ServingRuntime.set_tracking`),
and then deploy the function ({py:meth}`~mlrun.projects.MlrunProject.deploy_function`).

### Invoke the model

Invoke the function with {py:meth}`~mlrun.runtimes.RemoteRuntime.invoke`; after invoking, you can see results.

### Register and deploy the model-monitoring function

Add the monitoring function to the project using {py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_function`. 
Then, deploy the function using {py:meth}`~mlrun.projects.MlrunProject.deploy_function`.

You can use the MLRun built-in class, `EvidentlyModelMonitoringApplicationBase`, to integrate [Evidently](https://github.com/evidentlyai/evidently) as an MLRun function and create MLRun artifacts.

### Invoke the model again

Monitoring uses datasets defined by the parameter `base_period`. Invoking the model a second time ensures that the 
data includes the full monitoring window.

### View model monitoring artifacts and drift
 
- [Model monitoring in the platform UI](./monitoring-models.html#model-monitoring-in-the-platform-ui) to see the context and the artifacts.
- [Model monitoring in the Grafana dashboards](./monitoring-models.html#model-monitoring-in-the-grafana-dashboards) to see full details on model monitoring.

### See also

- {ref}`monitoring-models`
- {ref}`batch_inference_overview`