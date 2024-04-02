(monitoring)=

# Model monitoring
```{note}
This is currently a beta feature. This version does not support batch monitoring.
```

Model monitoring uses monitoring apps that are run on a set of model end-points. Each app returns either results or a list of results; the results are 
recorded in the DB.
The default app is a [batch monitoring app](https://www.mlrun.org/hub/functions/master/model-monitoring-batch/). 
You can use a pre-defined app or create your own. MLRun includes an app for integrating [Evidently](https://www.evidentlyai.com/) 
as an MLRun function to create MLRun artifacts (**add xref**).
You can also 


Model monitoring deploys three components:
- application controller function: handles the monitoring processing and the triggering applications. The controller 
sends the results to the app. 
- writer function: writes all of the monitoring application results to the databases. The writer is, in fact, 
- stream function: monitors the log of the data stream. It is triggered when a new log entry is detected. 
It processes the new events into statistics that are then written to statistics databases.

Drift is detected when it is identified by the threshold.
Scheduling is based on the job schedules.

When you call the {ref}`batch inference <batch_inference_overview>` job (stored in the [function hub](https://www.mlrun.org/hub/functions/master/batch_inference/)), 
you can specify the endpoint_id to apply the job to a specific existing model endpoint. 

Use `get_or_create_model_endpoint` to create a model endpoint record. Running `get_or_create_model_endpoint` 
also triggers writing new events to the Parquet target, and triggering the monitoring batch job for that new model endpoint.
If you don't provide an endpoint, the Parquet file is generated in a default context for the batch inference job (based on hashing 
of the batch inference job name in a particular project).




The first step is to enable model monitoring over your system with: {py:meth}`~mlrun.projects.MlrunProject.enable_model_monitoring`
The controller runs, by default, every 10 minutes, which is also the minimum interval. The interval is set per system, 
and not per function.You can modify the frequency with the parameter `base_period`. 
To change the `base_period`, first run `disable_model_monitoring`, then run `enable_model_monitoring`. 
The minium interval for scheduld jobs is 10 minutes, and can run in intervals of 10 (minutes).




Using an app 
ML-4218 v1.6
Adding the ability for a user to define an monitoring app that should be run on a set of model end-points. 

Within this phase there are a few restrictions that will be addressed on subsequent phase:  

- All apps will be running on all model endpoints 
- Scheduling is based on the job schedules. There is no "monitoring policy" yet
- There is a single job - dividing to several jobs by different sched/app/others will be done on the next phase. 
- Drift type will remain as is - drift is detected when it is identified by the threshold

ml-4620 - 1.6

Productization of model monitoring apps : 
- KV/TS 
- SDK APIs
- Batch monitoring as a default app 
- Sharding, scale. 
- ***Demoing with LLM ( model monitoring app - could be a model) ***


ml-4088 1.5
As part of implementing the model monitoring batch inference process within MLRun (right now it's defined in marketplace - 
https://www.mlrun.org/hub/functions/master/batch_inference/ ), we will expose an API for triggering this job. 
Please note that as part of the new API, the user can pass endpoint_id for applying the job to a specific existing 
model endpoint (and as a result to store the new events in the Parquet target of that endpoint). If not provided, 
the parquet file will be generated in a default context for the batch inference job. 
ml-4173
When calling monitoring batch inference job (stored in marketplace - https://www.mlrun.org/hub/functions/master/batch_inference/), 
the user will be able pass endpoint_id for applying the job to a specific existing model endpoint (
and as a result to store the new events in the Parquet target of that endpoint). If not provided, 
the parquet file will be generated in a default context for the batch inference job (based on hashing 
of the batch inference job name in a particular project). 

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

- {py:meth}`~mlrun.projects.MlrunProject.create_model_monitoring_function`
- {py:meth}`~mlrun.db.httpdb.HTTPRunDB.enable_model_monitoring` brings up controller and writer.
- mlrun enable project
- {py:meth}`~mlrun.projects.MlrunProject.disable_model_monitoring` - cancels the controller (not writer). In 1.6 controller is a job that runs /time. I 
- {py:meth}`~mlrun.projects.MlrunProject.list_model_monitoring_functions`
- {py:meth}`~mlrun.projects.MlrunProject.remove_model_monitoring_function`
- {py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_credentials` not new. Kafka or V3IO. will change in 1.7. Not sure if it works in 1.6.
- {py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_function`







## Configuring

enable monitoring -  base period is important - how often the controller runs (how often the entire app runs). 
How ofter the monitoring runs. Uses data from the base_period.
set_tracking on the model server on the serving function (same as before)
{py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_function` (=set and create) - and after deploy

(`create` creates a function but does not run it. Good for troubleshooting. Does not register the function to the project)

