(mm-running-applications)=

# Running the model monitoring application

This page explains how to run your model monitoring application on demand, as a "batch"
application, using MLRun's `evaluate` and `to_job` methods on existing model endpoint data.
This allows you to execute your application's monitoring logic as an MLRun job, either
locally or remotely, with flexible input and configuration options, including writing the
outputs to the databases.

If you want the application to run automatically ("real-time" application), use the standard
flow described in {ref}`register-model-monitoring-app`.

```{admonition} Note
You can also import model monitoring applications from the [MLRun hub](https://www.mlrun.org/hub/). Each application has complete usage instructions.
```

**In this section**

- [Overview](#overview)
- [Usage](#usage)

## Overview

The relevant methods of {py:class}`mlrun.model_monitoring.applications.ModelMonitoringApplicationBase` are:

- {py:meth}`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase.evaluate`
- {py:meth}`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase.to_job`

The `evaluate` method creates an MLRun job using the `to_job` method and runs it.
When more control over the job specifications is needed, you can use the `to_job` method
directly to customize the job configuration.

After testing your application with external data, as described in {ref}`testing-application-evaluate`,
you can run it on the actual model endpoint data and write the outputs with `write_output=True`.

## Usage

First, list the model endpoints and choose the ones you want to monitor:

```py
import mlrun

project = mlrun.get_or_create_project("my-project")
model_endpoints = project.list_model_endpoints(tsdb_metrics=True).endpoints
model_endpoint = model_endpoints[0]
```

Choose the start and end time. This example uses the first two hours of the data.
Since the start time is not inclusive, subtract a small `timedelta` from the first
request to include the start time in the data for the batch application run.

```py
from datetime import timedelta

start_time = model_endpoint.status.first_request - timedelta(microseconds=10)
end_time = model_endpoint.status.first_request + timedelta(hours=2)
```

Using the `evaluate` method directly is the simpler option:

```py
batch_app_run = ModelMonitoringApplicationBase.evaluate(
    class_handler="MyAppClass",
    func_path="src/my_application.py",
    func_name="monitoring-app-batch",
    endpoints=[(model_endpoint.metadata.name, model_endpoint.metadata.uid)],
    start=start_time,
    end=end_time,
    write_output=True,  # Write the outputs to the databases
    run_local=False,
)
```

Sometimes, modifications to the job are needed.
Get a job from the model monitoring application:

```py
from mlrun.model_monitoring.applications import ModelMonitoringApplicationBase

batch_app_job = ModelMonitoringApplicationBase.to_job(
    class_handler="MyAppClass",
    func_path="src/my_application.py",
    func_name="monitoring-app-batch",
)

# Modify the job specifications as needed
```

And run the job:

```py
batch_app_run = batch_app_job.run(
    params={
        "endpoints": [(model_endpoint.metadata.name, model_endpoint.metadata.uid)],
        "start": str(start_time),
        "end": str(end_time),
        "write_output": True,  # Write the outputs to the databases
    },
    local=False,
)
```

You can divide the run into small time windows with the `base_period` parameter. When used, the
difference between the `start` and `end` times is divided into smaller non-overlapping intervals,
each `base_period` minutes length.

When the model endpoint does not have data in the requested time window, the application will not
run.

### Running locally

When running locally and writing outputs (with `run_local=True` and `write_output=True`),
the `stream_profile` is required.
The stream profile is the datastore profile you already registered for the project.
See {ref}`mm-tsdb-streaming-platforms`.

### Overriding written data

Use the `existing_data_handling` parameter to control how the application handles existing data
in the output databases. By default, no time window overlap is allowed - `"fail_on_overlap"`.

If you want to override the existing data, use the `"delete_all"` value. It removes all the
data written by the application (identified with `func_name`) for the specified model endpoints.

The `"skip_overlap"` value allows to pass potential overlaps, but with a later start time for the new
data (ignoring the overlap data) so that it coincides with the `start` time of the already written data.
