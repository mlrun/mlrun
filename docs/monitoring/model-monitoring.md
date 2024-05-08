(model-monitoring)=
# Model monitoring user flow

This page gives an overview of the model monitoring feature. See a complete example in the tutorial [Model monitoring and drift detection](../tutorials/05-model-monitoring.html).
```{admonition} Note
If you are using the CE version, see {ref}`legacy-model-monitoring`.
```
## Architecture

<img src="../_static/images/model-monitoring.png" width="1100" >

The model monitoring process flow starts with collecting operational data. The operational data is converted to vectors, which are posted to the Model Server. 
The model server is then wrapped around a machine learning model that uses a function to calculate predictions based on the available vectors. Next, the model 
server creates a log for the input and output of the vectors, and the entries are written to the production data stream. While the model server 
is processing the vectors, the stream function monitors the log of the data stream and is triggered when a new log entry is detected. The stream function examines 
the log entry, processes it into statistics which are then written to the statistics databases (parquet file, time series database and key value database). 
The parquet files are written as a feature set under the model monitoring project. The parquet files can be read either using `pandas.read_parquet` or `feature_set.get_offline_features`, 
like any other feature set. In parallel, an MLRun job runs, reading the parquet files and performing drift analysis. The drift analysis data is stored so 
that the user can retrieve it in the Iguazio UI or in a Grafana dashboard.

When you enable model monitoring, you effectively deploy three components:
- application controller function: handles the monitoring processing and the triggers the apps that trigger the writer. The controller is a scheduled batch job whose frequency is determined by `base_period`. 
- stream function: monitors the log of the data stream. It is triggered when a new log entry is detected. The monitored data is used to create real-time dashboards, detect drift, and analyze performance.
- writer function: writes to the database and outputs alerts.

## APIs

The model monitoring APIs are configured per project. The APIs are:

- {py:meth}`~mlrun.projects.MlrunProject.enable_model_monitoring` &mdash; brings up the controller and schedules it according to the `base_period`; deploys the writer.
- {py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_function` &mdash; Update or set a monitoring function to the project. (Monitoring does not start until the function is deployed.) 
- {py:meth}`~mlrun.projects.MlrunProject.create_model_monitoring_function` &mdash; creates a function but does not set it. It's useful for troubleshooting, since it does  not register the function to the project.
- {py:meth}`~mlrun.projects.MlrunProject.list_model_monitoring_functions` &mdash; Retrieves a list of all the model monitoring functions.
- {py:meth}`~mlrun.projects.MlrunProject.remove_model_monitoring_function` &mdash; Removes the specified model-monitoring-app function from the project and from the DB.
- {py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_credentials` &mdash; Sets the Kafka or SQL credentials to be used by the project's model monitoring infrastructure functions. 
- {py:meth}`~mlrun.projects.MlrunProject.disable_model_monitoring` &mdash; disables the controller. 

## Configuration flow

### Enable model monitoring

Enable model monitoring for a project with {py:meth}`~mlrun.projects.MlrunProject.enable_model_monitoring`.
The controller runs, by default, every 10 minutes, which is also the minimum interval. 
You can modify the frequency with the parameter `base_period`. 
To change the `base_period`, first run `disable_model_monitoring`, then run `enable_model_monitoring`  
with the new `base_period` value. 

```python
project.enable_model_monitoring(base_period=1)
```

### Log the model with training data

See the parameter descriptions in {py:meth}`~mlrun.projects.MlrunProject.log_model`.
```python
model_name = "RandomForestClassifier"
project.log_model(
    model_name,
    model_file="./assets/model.pkl",
    training_set=train_set,
    framework="sklearn",
```

### Import, enable monitoring, and deploy the serving function

Use the [v2_model_server serving](https://www.mlrun.org/hub/functions/master/v2-model-server/) function 
from the MLRun function hub.

Add the model to the serving function's routing spec ({py:meth}`~mlrun.runtimes.ServingRuntime.add_model`), 
enable monitoring on the serving function ({py:meth}`~mlrun.runtimes.ServingRuntime.set_tracking`),
and then deploy the function ({py:meth}`~mlrun.projects.MlrunProject.deploy_function`).
```python
# Import the serving function
serving_fn = import_function(
    "hub://v2_model_server", project=project_name, new_name="serving"
)

serving_fn.add_model(
    model_name, model_path=f"store://models/{project_name}/{model_name}:latest"
)

# enable monitoring on this serving function
serving_fn.set_tracking()

serving_fn.spec.build.requirements = ["scikit-learn"]

# Deploy the serving function
project.deploy_function(serving_fn)
```

### Invoke the model

Invoke the model function with {py:meth}`~mlrun.runtimes.RemoteRuntime.invoke`.


```python
model_name = "RandomForestClassifier"
serving_1 = project.get_function("serving")
0
for i in range(150):
    # data_point = choice(iris_data)
    data_point = [0.5, 0.5, 0.5, 0.5]
    serving_1.invoke(
        f"v2/models/{model_name}/infer", json.dumps({"inputs": [data_point]})
    )
    sleep(choice([0.01, 0.04]))
```
After invoking the model, you can see some basic meta data, including last prediction and average latency.

<img src="../tutorials/_static/images/model_endpoint_1.png" width="1000" >

You can also see the basic statistics in Grafana.

### Register and deploy the model-monitoring function

Add the monitoring function to the project using {py:meth}`~mlrun.projects.MlrunProject.set_model_monitoring_function`. 
Then, deploy the function using {py:meth}`~mlrun.projects.MlrunProject.deploy_function`.

You can use the MLRun built-in class, `EvidentlyModelMonitoringApplicationBase`, to integrate [Evidently](https://github.com/evidentlyai/evidently) as an MLRun function and create MLRun artifacts.
```python
my_app = project.set_model_monitoring_function(
    func="./assets/demo_app.py",
    application_class="DemoMonitoringApp",
    name="myApp",
)

project.deploy_function(my_app)
```
### Invoke the model again

Monitoring uses datasets defined by the parameter `base_period`. Invoking the model a second time ensures that the 
data set includes the full monitoring window. From this point on, the controller checks the Parquet DB every 10 minutes (or non-default 
`base_period`) and streams any new data to the app.
```python
model_name = "RandomForestClassifier"
serving_1 = project.get_function("serving")

for i in range(150):
    data_point = choice(iris_data)
    # data_point = [0.5,0.5,0.5,0.5]
    serving_1.invoke(
        f"v2/models/{model_name}/infer", json.dumps({"inputs": [data_point]})
    )
    sleep(choice([0.01, 0.04]))
```

### View model monitoring artifacts and drift
 
- [Model monitoring in the platform UI](./monitoring-models.html#model-monitoring-in-the-platform-ui) to see the context and the artifacts.
- [Model monitoring in the Grafana dashboards](./monitoring-models.html#model-monitoring-in-the-grafana-dashboards) to see full details on model monitoring.

### See also

- {ref}`monitoring-models`
- {ref}`batch_inference_overview`