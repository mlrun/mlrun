(monitoring)=
# Model monitoring
```{note}
This is currently a beta feature. 
```
This page gves an overview of the model monitoring feature. See a complete example in the {ref}`model=monitor-drift-detect` tutorial.

## Architecture

new diagram


When you enable model monitoring, you effectively deploy three components:
- application controller function: handles the monitoring processing and the triggering applications. The controller 
sends the results to the app. 
- writer function: writes all of the monitoring application results to the databases. The writer is, in fact, 
- stream function: monitors the log of the data stream. It is triggered when a new log entry is detected. The monitored data is used to create real-time dashboards, detect drift, and analyze performance.

Model monitoring is run on a set of model end-points. Each job returns either results or a 
list of results; the results are recorded in the DB.






You can also 







Scheduling **of what??** is based on the particular job's schedule  **Meaning what?  job.run(schedule="0 * * * *")**


in v1.6?
Use `get_or_create_model_endpoint` to create a model endpoint record. Running `get_or_create_model_endpoint` 
also triggers writing new events to the Parquet target, and triggering the monitoring batch job for that new model endpoint.
If you don't provide an endpoint, the Parquet file is generated in a default context for the batch inference job (based on hashing 
of the batch inference job name in a particular project).

## APIs

The model monitoring APIs:

- {py:meth}`~mlrun.db.httpdb.HTTPRunDB.enable_model_monitoring` &mdash;brings up controller and writer.
- {py:meth}`~mlrun.projects.MlrunProject.disable_model_monitoring` &mdash; cancels the controller (not writer). 
- {py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_function` &mdash; Update or add a monitoring function to the project.  
- {py:meth}`~mlrun.projects.MlrunProject.create_model_monitoring_function` &mdash; creates a function but does not run it. It's useful for troubleshooting, since it does  not register the function to the project.
- {py:meth}`~mlrun.projects.MlrunProject.list_model_monitoring_functions` &mdash; Retrieve a list of all the model monitoring functions.
- {py:meth}`~mlrun.projects.MlrunProject.remove_model_monitoring_function` &mdash; Removes the specified model-monitoring-app function from the project and from the DB
- {py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_credentials` not new. Kafka or V3IO. will change in 1.7. Not sure if it works in 1.6.

## Configuration flow

The general configuration flow:
### Enable model monitoring
`
Enable model monitoring over your system with {py:meth}`~mlrun.projects.MlrunProject.enable_model_monitoring`.
Model monitoring is enabled across the system (and not the project). `enable_model_monitoring` deploys the 
controller, stream, and writer, infrastructure pods, and creates the connecting streams.

The controller runs, by default, every 10 minutes, which is also the minimum interval. The interval is set per system 
(and not per function). You can modify the frequency with the parameter `base_period`. 
To change the `base_period`, first run `disable_model_monitoring`, then run `enable_model_monitoring` and 
set the new `base_period` value. **Is update_model_monitoring_controller only from v1.7?**

### Log the model with training data
{py:meth}`~mlrun.projects.MlrunProject.log_model`
Log the model using the project API so that it is available through the feature store API.
**WHY do I need the FS API?**

### Import, enable monitoring, and deploy the serving function

Add the model to the serving function's routing spec ({py:meth}`~mlrun.runtimes.ServingRuntime.add_model`), 
enable monitoring on the serving function ({py:meth}`~smlrun.runtimes.ServingRuntime.set_tracking`)
and then deploy the function ({py:meth}`~mlrun.projects.MlrunProject.deploy_function`)

You can use the [v2_model_server serving](https://www.mlrun.org/hub/functions/master/v2-model-server/) function from the MLRun function hub,
or alternatively, another serving function.

### Invoke the model

Invoke the function and return the results ({py:meth}`~mlrun.runtimes.RemoteRuntime.invoke`).

### Register and deploy the model-monitoring function

Add the monitoring function to the project. **Is this a default: ({py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_function`). **
Then, deploy the function ({py:meth}`~mlrun.runtimes.RemoteRuntime.deploy`).

**Says in API page: fn.deploy() where fn is the object returned by this method., which does not make sense**

You can use a pre-defined function or create your own. MLRun includes a function for integrating [Evidently](https://www.evidentlyai.com/) 
as an MLRun function to create MLRun artifacts (**add xref**). **Where is it located?**

### Invoke the model again

to ensure that the monitoring window closed.
**Is this what gives something to copare to?**

### Run the batch inference

When you call the {ref}`batch inference <batch_inference_overview>` function (stored in the [function hub](https://www.mlrun.org/hub/functions/master/batch_inference_2/)), 
you can specify the endpoint_id &mdash; to apply the job to a specific existing model endpoint. 

See more about [batch inference](../deployment/batch_inference.html).

### View artifacts and drift
 
   - UI: Since model monitoring is implemented with functions, Jobs and Workflows > Monitor Jobs shows details about all of the jobs that comprise the model mointoring.
   - The "Models" give ***?*
   - Grafana has [dashboards](./monitoring-models.html#model-monitoring-in-the-grafana-dashboards) that provide full details on model monitoring.

### Add monitoring to a deployed model
**???**

Apply `set_tracking()` on your serving function and specify the function spec attributes:

        fn.set_tracking(stream_path, batch, sample)

* `stream_path` &mdash; Enterprise: the v3io stream path (e.g. `v3io:///users/..`); CE: a valid Kafka stream 
(e.g. kafka://kafka.default.svc.cluster.local:9092)
* `sample` &mdash; optional, sample every N requests
* `batch` &mdash; optional, send micro-batches every N requests
* `tracking_policy` &mdash; optional, model tracking configurations, such as setting the scheduling policy of the model monitoring batch job




- ***Demoing with LLM ( model monitoring app - could be a model) ***



**START HERE**

mlrun/api/api/endpoints/jobs.py - New api for submitting specific jobs. At the moment, it only includes 
the model monitoring batch job.
Model Monitoring Batch Job - This job can process a list of specific model endpoints
Model Monitoring Feature Set - Model endpoint id is now added as a tag on each feature set. The main reason 
for that is that now the user can define his own endpoint id but the feature set is still being 
generated based on the function and model name. Therefore, using the endpoint id as a tag, we 
can avoid overlapping of feature sets.
++++++++++++++++++++++++++++++++++++++++++

Need an Apps - ModelMonitoringApplicationBase or EvidentlyModelMonitoringApplicationBase
user can write init function eg to include directory, 
need `do_tracking` (= function)
self.context - gives MLRun context incl artifacts - context does appear in UI under Jobs and workflows.

ModelMonitoringApplicationResult - 
name - is saved in KV and TSDB
class ResultKindApp(Enum):
    """
    Enum for the result kind values
    """

    data_drift = 0
    concept_drift = 1
    model_performance = 2
    system_performance = 3


class ResultStatusApp(IntEnum):
    """
    Enum for the result status values, detected means that the app detected some problem.
    """

    irrelevant = -1
    no_detection = 0
    potential_detection = 1
    detected = 2
	
extra_data:  whatever data user wants to save.	
	

![Architecture](../_static/images/model-monitoring.png)


