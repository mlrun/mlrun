(mm-applications)=
# Writing a model monitoring application

Learn how to create your own model monitoring applications for LLMs, gen AI, deep-learning models, etc., based on the `ModelMonitoringApplicationBase` class.

**In this section**
- [Basics](#basics)
- [Using the application context](#using-the-application-context)
- [Testing your application  before deploying it](#testing-your-application-before-deploying-it)
- [Evidently-based application](#evidently-based-application)

## Basics

First, create a Python module and import the API objects:

```py
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationMetric,
    ModelMonitoringApplicationResult,
    MonitoringApplicationContext,
)
from mlrun.common.schemas.model_monitoring import ResultKindApp, ResultStatusApp
```

Then, write the application itself by inheriting from the {py:class}`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase` class. You have to implement the do_tracking` method.
Here is a "dummy" app that returns a constant result for each monitoring window:

```py
class ServingMonitoring(ModelMonitoringApplicationBase):
    def do_tracking(
        self, monitoring_context: MonitoringApplicationContext
    ) -> ModelMonitoringApplicationResult:
        return ModelMonitoringApplicationResult(
            name="dummy-res-just-for-demo",
            value=0,
            status=ResultStatusApp.irrelevant,
            kind=ResultKindApp.mm_app_anomaly,
        )
```

The `do_tracking` method of the application object is called for each "closed" monitoring time window
of each monitored model-endpoint and returns a result.
The result may be just one result, as in the example above, or a list of results
({py:class}`~mlrun.model_monitoring.applications.ModelMonitoringApplicationResult`) and metrics ({py:class}`~mlrun.model_monitoring.applications.ModelMonitoringApplicationMetric`).

The application class may implement a custom `__init__` constructor with arguments.

To register and deploy the application see {ref}`register-model-monitoring-app`.

(testing-application-evaluate)=
## Testing your application before deploying it

You can run and debug your application as a job with data, but without a model endpoint or datastore profiles. This reduces
the time required to refine your model before deploying.
The monitoring creates metrics that assist you in understanding and refining the model behavior.
You can use this flow for both local and remote jobs.

Use {py:meth}`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase.evaluate` to test your code.
When you are satisfied with the application, deploy it with {py:meth}`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase.deploy`.

For example, import the source file:

```py
# Myapp.py
import mlrun
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationMetric,
    ModelMonitoringApplicationResult,
)


class MyApp(ModelMonitoringApplicationBase):
    """User code"""

    def do_tracking(self, monitoring_context):
        print(monitoring_context.__dict__)
        results = [
            ModelMonitoringApplicationMetric(name="test_metric", value=0.1),
            ModelMonitoringApplicationResult(
                name="test_result",
                value=0.2,
                kind=mlrun.common.schemas.model_monitoring.constants.ResultKindApp.system_performance,
                status=mlrun.common.schemas.model_monitoring.constants.ResultStatusApp.no_detection,
            ),
        ]
        return results
```

Then, import the class and run `evaluate`.

```py
from Myapp import MyApp

MyApp.evaluate(
    func_path="Myapp.py",
    run_local=False,
    sample_data=pd.DataFrame({"col": [1, 2, 3, 4]}),
)
```

After you have fine-tuned the model monitoring application, deploy it with:

```py
MyApp.deploy(
    func_path="Myapp.py",
    func_name="run-me-in-wf",
)
```

## Using the application context

The `monitoring_context` argument is a
{py:class}`~mlrun.model_monitoring.applications.context.MonitoringApplicationContext` object.
It includes the current window data as a pandas data-frame: `monitoring_context.sample_df`.
The reference and current data is also available in raw format as `monitoring_context.feature_stats`
and `monitoring_context.sample_df_stats`, respectively.

The `monitoring_context` provides also attributes and methods to log application messages or artifacts.

Logging a debug message:

```py
monitoring_context.logger.debug(
    "Logging the current data of a specific endpoint",
    sample_df=context.sample_df.to_json(),
    endpoint_id=context.endpoint_id,
)
```

Logging an artifact:

```py
monitoring_context.log_artifact(
    item=f"num_events_last_monitoring_window_{context.endpoint_id}",
    body=f"Number of events in the window: {len(context.sample_df)}",
)
```

```{caution}
Logging artifacts in every model monitoring window may cause scale issues.

The `log_artifact` and `log_dataset` methods of the `monitoring_context` should be called on special occasions only.
<!-- ML-9550, ML-7677 -->
```

## Evidently-based application

To create an Evidently based model monitoring application, import the following class:

```py
from mlrun.model_monitoring.applications.evidently import (
    EvidentlyModelMonitoringApplicationBase,
)
```

Inherit from it, implement the `do_tracking` method, and pass the `evidently_workspace_path` and
`evidently_project_id` arguments upon construction.

```{caution}
Evidently has a memory accumulation [issue](https://github.com/evidentlyai/evidently/issues/1217)
as more and more snapshots are saved.

The method `log_project_dashboard` should be called on special occasions only, as well as
saving Evidently project snapshots through `project.add_snapshot`.
<!-- ML-7159 -->
```

To add the `evidently` package to the model monitoring application image:

```py
project.set_model_monitoring_function(
    # Set the required arguments
    requirements=["evidently"],
)
```

```{note}
It is recommended to specify the exact version of the `evidently` package for reproducibility with
`"evidently==<x.y.z>"`. Get the supported version through
`from mlrun.model_monitoring.applications.evidently import SUPPORTED_EVIDENTLY_VERSION`.
```

See a full example in [Realtime monitoring and drift detection](../tutorials/05-model-monitoring.ipynb#deploying-evidently-based-app).

