(run_project_functions)=
# Run, build, and deploy functions

**In this section**
- [Overview](#overview)
- [run_function](#run)
- [build_function](#build)
- [deploy_function](#deploy)

<a id="overview"></a>
## Overview

There is a set of methods used to deploy and run project functions. They can be used interactively or inside a pipeline (e.g. Kubeflow). 
When used inside a pipeline, each method is automatically mapped to the relevant pipeline engine command.

* {py:meth}`~mlrun.projects.run_function` &mdash; Run a local or remote task as part of local or remote batch/scheduled task
* {py:meth}`~mlrun.projects.build_function` &mdash; deploy an ML function, build a container with its dependencies for use in runs
* {py:meth}`~mlrun.projects.deploy_function` &mdash; deploy real-time/online (nuclio or serving based) functions

You can use those methods as `project` methods, or as global (`mlrun.`) methods. For example:

    # run the "train" function in myproject
    run = myproject.run_function("train", inputs={"data": data_url})  
    
    # run the "train" function in the current/active project (or in a pipeline)
    run = mlrun.run_function("train", inputs={"data": data_url})  
    
The first parameter in all three methods is either the function name (in the project), or a function object, used if you want to 
specify functions that you imported/created ad hoc, or to modify a function spec. For example:

    # import a serving function from the Function Hub and deploy a trained model over it
    serving = import_function("hub://v2_model_server", new_name="serving")
    serving.spec.replicas = 2
    deploy = deploy_function(
        serving,
        models=[{"key": "mymodel", "model_path": train.outputs["model"]}],
    )
    
You can use the {py:meth}`~mlrun.projects.MlrunProject.get_function` method to get the function object and manipulate it, for example:

    trainer = project.get_function("train")
    trainer.with_limits(mem="2G", cpu=2, gpus=1)
    run = project.run_function("train", inputs={"data": data_url}) 


<a id="run"></a>
## run_function

Use the {py:meth}`~mlrun.projects.run_function` method to run a local or remote batch/scheduled task.
The `run_function` method accepts various parameters such as `name`, `handler`, `params`, `inputs`, `schedule`, etc. 
Alternatively, you can pass a **`Task`** object (see: {py:func}`~mlrun.model.new_task`) that holds all of the 
parameters and the advanced options. 

Functions can host multiple methods (handlers). You can set the default handler per function. You need to specify which handler you intend to call in the run command. 
You can pass `parameters` (arguments) or data `inputs` (such as datasets, feature-vectors, models, or files) to the functions through the `run_function` method.
 
The {py:meth}`~mlrun.projects.run_function` command returns an MLRun {py:class}`~mlrun.model.RunObject` object that you can use to track the job and its results. 
If you pass the parameter `watch=True` (default), the command blocks until the job completes.

MLRun also supports iterative jobs that can run and track multiple child jobs (for hyperparameter tasks, AutoML, etc.). 
See {ref}`hyper-params` for details and examples.

Read further details on [**running tasks and getting their results**](../concepts/submitting-tasks-jobs-to-functions.html).

Usage examples:

    # create a project with two functions (local and from Function Hub)
    project = mlrun.new_project(project_name, "./proj")
    project.set_function("mycode.py", "prep", image="mlrun/mlrun")
    project.set_function("hub://auto_trainer", "train")

    # run functions (refer to them by name)
    run1 = project.run_function("prep", params={"x": 7}, inputs={'data': data_url})
    run2 = project.run_function("train", inputs={"dataset": run1.outputs["data"]})
    run2.artifact('confusion-matrix').show()


```{admonition} Run/simulate functions locally: 
Functions can also run and be debugged locally by using the `local` runtime or by setting the `local=True` 
parameter in the {py:meth}`~mlrun.runtimes.BaseRuntime.run` method (for batch functions).
```

<a id="build"></a>
## build_function

The {py:meth}`~mlrun.projects.build_function` method is used to deploy an ML function and build a container with its dependencies for use in runs.

Example:

    # build the "trainer" function image (based on the specified requirements and code repo)
    project.build_function("trainer")

The {py:meth}`~mlrun.projects.build_function` method accepts different parameters that can add to, or override, the function build spec.
You can specify the target or base `image` extra docker `commands`, builder environment, and source credentials (`builder_env`), etc. 

See further details and examples in [**Build function image**](../runtimes/image-build.html). 


<a id="deploy"></a>
## deploy_function

The {py:meth}`~mlrun.projects.deploy_function` method is used to deploy real-time/online (nuclio or serving) functions and pipelines.
Read more about [**Real-time serving pipelines**](../serving/serving-graph.html).

Basic example:

    # Deploy a real-time nuclio function ("myapi")
    deployment = project.deploy_function("myapi")
    
    # invoke the deployed function (using HTTP request) 
    resp = deployment.function.invoke("/do")

You can provide the `env` dict with: extra environment variables; `models` list to specify specific models and their attributes 
(in the case of serving functions); builder environment; and source credentials (`builder_env`).

Example of using `deploy_function` inside a pipeline, after the `train` step, to generate a model:

    # Deploy the trained model (from the "train" step) as a serverless serving function
    serving_fn = mlrun.new_function("serving", image="mlrun/mlrun", kind="serving")
    mlrun.deploy_function(
        serving_fn,
        models=[
            {
                "key": model_name,
                "model_path": train.outputs["model"],
                "class_name": 'mlrun.frameworks.sklearn.SklearnModelServer',
            }
        ],
    )


```{admonition} Note
If the `mock` flag is set to `True`, MLRun creates a simulated (mock) function instead of a real Kubernetes service.
```