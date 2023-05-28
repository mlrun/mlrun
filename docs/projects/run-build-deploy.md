(run_project_functions)=
# Run, build, and deploy functions

**In this section**
- [Overview](#overview)
- [run_function](#run)
- [build_function](#build)
- [deploy_function](#deploy)
- [Default image](#default_image)
- [Image build configuration](#build_config)
- [build_image](#build_image)

<a id="overview"></a>
## Overview

There is a set of methods used to deploy and run project functions. They can be used interactively or inside a pipeline (e.g. Kubeflow). 
When used inside a pipeline, each method is automatically mapped to the relevant pipeline engine command.

* {py:meth}`~mlrun.projects.run_function` &mdash; Run a local or remote task as part of local or remote batch/scheduled task
* {py:meth}`~mlrun.projects.build_function` &mdash; deploy an ML function, build a container with its dependencies for use in runs
* {py:meth}`~mlrun.projects.deploy_function` &mdash; deploy real-time/online (nuclio or serving based) functions

You can use those methods as `project` methods, or as global (`mlrun.`) methods. For example:

```python
# run the "train" function in myproject
run = myproject.run_function("train", inputs={"data": data_url})  

# run the "train" function in the current/active project (or in a pipeline)
run = mlrun.run_function("train", inputs={"data": data_url})
```
    
The first parameter in all three methods is either the function name (in the project), or a function object, used if you want to 
specify functions that you imported/created ad hoc, or to modify a function spec. For example:

```python
# import a serving function from the Function Hub and deploy a trained model over it
serving = import_function("hub://v2_model_server", new_name="serving")
serving.spec.replicas = 2
deploy = deploy_function(
  serving,
  models=[{"key": "mymodel", "model_path": train.outputs["model"]}],
)
```
    
You can use the {py:meth}`~mlrun.projects.MlrunProject.get_function` method to get the function object and manipulate it, for example:

```python
trainer = project.get_function("train")
trainer.with_limits(mem="2G", cpu=2, gpus=1)
run = project.run_function("train", inputs={"data": data_url}) 
```

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

```python
# create a project with two functions (local and from Function Hub)
project = mlrun.new_project(project_name, "./proj")
project.set_function("mycode.py", "prep", image="mlrun/mlrun")
project.set_function("hub://auto_trainer", "train")

# run functions (refer to them by name)
run1 = project.run_function("prep", params={"x": 7}, inputs={'data': data_url})
run2 = project.run_function("train", inputs={"dataset": run1.outputs["data"]})
run2.artifact('confusion-matrix').show()
```

```{admonition} Run/simulate functions locally: 
Functions can also run and be debugged locally by using the `local` runtime or by setting the `local=True` 
parameter in the {py:meth}`~mlrun.runtimes.BaseRuntime.run` method (for batch functions).
```

<a id="build"></a>
## build_function

The {py:meth}`~mlrun.projects.build_function` method is used to deploy an ML function and build a container with its dependencies for use in runs.

Example:

```python
# build the "trainer" function image (based on the specified requirements and code repo)
project.build_function("trainer")
```

The {py:meth}`~mlrun.projects.build_function` method accepts different parameters that can add to, or override, the function build spec.
You can specify the target or base `image` extra docker `commands`, builder environment, and source credentials (`builder_env`), etc. 

See further details and examples in [**Build function image**](../runtimes/image-build.html). 


<a id="deploy"></a>
## deploy_function

The {py:meth}`~mlrun.projects.deploy_function` method is used to deploy real-time/online (nuclio or serving) functions and pipelines.
Read more about [**Real-time serving pipelines**](../serving/serving-graph.html).

Basic example:

```python
# Deploy a real-time nuclio function ("myapi")
deployment = project.deploy_function("myapi")

# invoke the deployed function (using HTTP request) 
resp = deployment.function.invoke("/do")
```

You can provide the `env` dict with: extra environment variables; `models` list to specify specific models and their attributes 
(in the case of serving functions); builder environment; and source credentials (`builder_env`).

Example of using `deploy_function` inside a pipeline, after the `train` step, to generate a model:

```python
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
```


```{admonition} Note
If you want to create a simulated (mock) function instead of a real Kubernetes service, set the `mock` flag is set to `True`. See [deploy_function api](https://docs.mlrun.org/en/latest/api/mlrun.projects.html#mlrun.projects.MlrunProject.deploy_function).
```

<a id="default_image"></a>
## Default image

You can set a default image for the project. This image will be used for deploying and running any function that does
not have an explicit image assigned, and replaces MLRun's default image of `mlrun/mlrun`. To set the default image use 
the {py:meth}`~mlrun.projects.MlrunProject.set_default_image` method with the name of the image.

The default image is applied to the functions in the process of enriching the function prior to running or 
deploying. Functions will therefore use the default image set in the project at the time of their execution, not the
image that was set when the function was added to the project.

For example:

```python
 project = mlrun.new_project(project_name, "./proj")
 # use v1 of a pre-built image as default
 project.set_default_image("myrepo/my-prebuilt-image:v1")
 # set function without an image, will use the project's default image
 project.set_function("mycode.py", "prep")

 # function will run with the "myrepo/my-prebuilt-image:v1" image
 run1 = project.run_function("prep", params={"x": 7}, inputs={'data': data_url})

 ...

 # replace the default image with a newer v2
 project.set_default_image("myrepo/my-prebuilt-image:v2")
 # function will now run using the v2 version of the image 
 run2 = project.run_function("prep", params={"x": 7}, inputs={'data': data_url})
```

<a id="build_config"></a>
## Image build configuration

Use the {py:meth}`~mlrun.projects.MlrunProject.set_default_image` function to configure a project to use an existing 
image. The configuration for building this default image can be contained within the project, by using the 
{py:meth}`~mlrun.projects.MlrunProject.build_config` and {py:meth}`~mlrun.projects.MlrunProject.build_image` 
functions. 

The project build configuration is maintained in the project object. When saving, exporting and importing the project 
these configurations are carried over with it. This makes it simple to transport a project between systems while 
ensuring that the needed runtime images are built and are ready for execution. 

When using {py:meth}`~mlrun.projects.MlrunProject.build_config`, build configurations can be passed along with the 
resulting image name, and these are used to build the image. The image name is assigned following these rules, 
based on the project configuration and provided parameters:

1. If provided, the name passed in the `image` parameter of {py:meth}`~mlrun.projects.MlrunProject.build_config`.
2. The project's default image name, if configured using {py:meth}`~mlrun.projects.MlrunProject.set_default_image`.
3. The value set in MLRun's `default_project_image_name` config parameter - by default this value is 
   `.mlrun-project-image-{name}` with the project name as template parameter.

For example:

```python
 # Set image config for current project object, using base mlrun image with additional requirements. 
 image_name = ".my-project-image"
 project.build_config(
     image=image_name,
     set_as_default=True,
     with_mlrun=False,
     base_image="mlrun/mlrun",
     requirements=["vaderSentiment"],
 )

 # Export the project configuration. The yaml file will contain the build configuration
 proj_file_path = "~/mlrun/my-project/project.yaml"
 project.export(proj_file_path)
```
 
This project can then be imported and the default image can be built:

```python
 # Import the project as a new project with a different name
 new_project = mlrun.load_project("~/mlrun/my-project", name="my-other-project")
 # Build the default image for the project, based on project build config
 new_project.build_image()

 # Set a new function and run it (new function uses the my-project-image image built previously)
 new_project.set_function("sentiment.py", name="scores", kind="job", handler="handler")
 new_project.run_function("scores")
```

<a id="build_image"></a>
## build_image

The {py:meth}`~mlrun.projects.MlrunProject.build_image` function builds an image using the existing build configuration. 
This method can also be used to set the build configuration and build the image based on it - in a single step. 

When using `set_as_default=False` any build config provided is still kept in the project object but the generated 
image name is not set as the default image for this project. 

For example:

```python
image_name = ".temporary-image"
project.build_image(image=image_name, set_as_default=False)

# Create a function using the temp image name
project.set_function("sentiment.py", name="scores", kind="job", handler="handler", image=image_name)
```
   