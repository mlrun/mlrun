(submitting-tasks-jobs-to-functions)=
#  Running a job

MLRun batch function objects support a {py:meth}`~mlrun.runtimes.BaseRuntime.run` method for invoking a job over them. 
The run method accepts various parameters such as `name`, `handler`, `params`, `inputs`, `schedule`, etc. 
Alternatively, you can pass a **`Task`** object (see: {py:func}`~mlrun.model.new_task`) that holds all of the 
parameters plus the advanced options. 

```{admonition} Run/simulate functions locally: 
Functions can also run and be debugged locally by using the `local` runtime or by setting the `local=True` 
parameter in the {py:meth}`~mlrun.runtimes.BaseRuntime.run` method (for batch functions).
```

Functions can host multiple methods (handlers). You can set the default handler per function. You
 need to specify which handler you intend to call in the run command. 

You can pass `parameters` (arguments) or data `inputs` (such as datasets, feature-vectors, models, or files) to the functions through the `run` method.
- Inside the function you can access the parameters/inputs by simply adding them as parameters to the function or you can get them from the context object (using `get_param()` and ` get_input()`).
- Various data objects (files, tables, models, etc.) are passed to the function as data item objects. You can pass data objects using the 
inputs dictionary argument, where the dictionary keys match the function's handler argument names and the MLRun data urls are provided 
as the values. The data is passed into the function as a {py:class}`~mlrun.datastore.DataItem` object that handles data movement, 
tracking and security in an optimal way. Read more about data objects in [Data stores](../store/datastore.html).


    run_results = fn.run(params={"label_column": "label"}, inputs={'data': data_url})

MLRun also supports iterative jobs that can run and track multiple child jobs (for hyperparameter tasks, AutoML, etc.). 
See {ref}`hyper-params` for details and examples.
 
The `run()` command returns a run object that you can use to track the job and its results. If you
pass the parameter `watch=True` (default) the {py:meth}`~mlrun.runtimes.BaseRuntime.run` command blocks 
until the job completes.

Run object has the following methods/properties:
- `uid()` &mdash; returns the unique ID.
- `state()` &mdash; returns the last known state.
- `show()` &mdash; shows the latest job state and data in a visual widget (with hyperlinks and hints).
- `outputs` &mdash; returns a dictionary of the run results and artifact paths.
- `logs(watch=True)` &mdash; returns the latest logs.
    Use `Watch=False` to disable the interactive mode in running jobs.
- `artifact(key)` &mdash; returns an artifact for the provided key (as {py:class}`~mlrun.datastore.DataItem` object).
- `output(key)` &mdash; returns a specific result or an artifact path for the provided key.
- `wait_for_completion()` &mdash; wait for async run to complete
- `refresh()` &mdash; refresh run state from the db/service
- `to_dict()`, `to_yaml()`, `to_json()` &mdash; converts the run object to a dictionary, YAML, or JSON format (respectively).

<br>You can view the job details, logs, and artifacts in the UI. When you first open the **Monitor 
Jobs** tab it displays the last jobs that ran and their data. Click a job name to view its run history, and click a run to view more of the 
run's data.

<br><img src="../_static/images/project-jobs-train-artifacts-test_set.png" alt="project-jobs-train-artifacts-test_set" width="800"/>

See full details and examples in [Functions](../runtimes/functions.html).