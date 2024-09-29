(mm-applications)=
# Writing a model monitoring application

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

Then, write the application itself by inheriting from the `ModelMonitoringApplicationBase` base class.
You have to implement the `do_tracking` method.
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
`ModelMonitoringApplicationResult` and metrics `ModelMonitoringApplicationMetric`.

The application class may implement a custom `__init__` constructor with arguments.

To register and deploy the application see {ref}`register-model-monitoring-app`.

## Using the application context

The `context` argument is a `MonitoringApplicationContext` object.
It includes the current window data as a pandas data-frame: `context.sample_df`.
The reference and current data is also available in raw format as `context.feature_stats`
and `context.sample_df_stats`, respectively.

The `context` provides also attributes and methods to log application messages or artifacts.

Logging a debug message:

```py
context.logger.debug(
    "Logging the current data of a specific endpoint",
    sample_df=context.sample_df.to_json(),
    endpoint_id=context.endpoint_id,
)
```

Logging an artifact:

```py
context.log_artifact(
    item=f"num_events_last_monitoring_window_{context.endpoint_id}",
    body=f"Number of events in the window: {len(context.sample_df)}",
)
```

```{caution}
Since each new artifact is saved in the artifact-manager store (in memory), it is not recommended
to store a new artifact on each application run. Instead you can:

- Override artifacts by using the same key.
- Save artifacts with a unique key in special occasions, e.g., when a drift is detected.

<!-- ML-7347 -->
```

## Evidently-based application

To create an Evidently based model monitoring application, import the following class:

```py
from mlrun.model_monitoring.applications import EvidentlyModelMonitoringApplicationBase
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
`mlrun.model_monitoring.evidently_application.SUPPORTED_EVIDENTLY_VERSION`.
```
