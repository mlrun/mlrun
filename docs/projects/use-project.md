(Using-projects)=
# Using projects

You can add/update a project's functions, artifacts, or workflows using {py:meth}`~mlrun.projects.MlrunProject.set_function`, 
{py:meth}`~mlrun.projects.MlrunProject.set_artifact`, {py:meth}`~mlrun.projects.MlrunProject.set_workflow`, and set
various project attributes (`parameters`, `secrets`, etc.).

Use the project {py:meth}`~mlrun.projects.MlrunProject.run` method to run a registered workflow using a pipeline engine (e.g. 
Kubeflow pipelines). The workflow executes its registered functions in a sequence/graph (DAG). The workflow can reference project
parameters, secrets, and artifacts by name.

Projects can also be loaded and workflows/pipelines can be executed using the CLI (using `mlrun project` command).

**In this section**
- [Updating and using project functions](#updating-and-using-project-functions)
- [Run, build, and deploy functions](#run-build-and-deploy-functions)


## Updating and using project functions

Projects host or link to functions that are used in job or workflow runs. You add functions to a project using 
{py:meth}`~mlrun.projects.MlrunProject.set_function`. This registers them as part of the project definition (and Yaml file).
Alternatively, you can create functions using methods like {py:func}`~mlrun.run.code_to_function` and save them to the DB (under the same project). 
The preferred approach is to use `set_function` (which also records the functions in the project spec).

The {py:meth}`~mlrun.projects.MlrunProject.set_function` method allow you to add/update many types of functions:
* **marketplace functions** - load/register a marketplace function into the project (func="hub://...")
* **notebook file** - convert a notebook file into a function (func="path/to/file.ipynb")
* **python file** - convert a python file into a function (func="path/to/file.py")
* **database function** - function stored in MLRun DB (func="db://project/func-name:version")
* **function yaml file** - read the function object from a yaml file (func="path/to/file.yaml")
* **inline function spec** - save the full function spec in the project definition file (func=func_object), not recommended

When loading a function from code file (py, ipynb) you should also specify a container `image` and the runtime `kind` (will use `job` kind as default).
You can optionally specify the function `handler` (the function handler to invoke), and a `name`.

If the function is not a single file function, and it requires access to multiple files/libraries in the project, 
you should set the `with_repo=True` to add the entire repo code into the destination container during build or run time.

```{admonition} Note
When using `with_repo=True` the functions need to be deployed (`function.deploy()`) to build a container, unless you set `project.spec.load_source_on_run=True` which instructs MLRun to load the git/archive repo into the function container 
at run time and do not require a build (this is simpler when developing, for production it's preferred to build the image with the code)
```

Examples:

```python
    project.set_function('hub://sklearn_classifier', 'train')
    project.set_function('http://.../mynb.ipynb', 'test', image="mlrun/mlrun")
    project.set_function('./src/mycode.py', 'ingest',
                         image='myrepo/ing:latest', with_repo=True)
    project.set_function('db://project/func-name:version')
    project.set_function('./func.yaml')
    project.set_function(func_object)
```
You can get the function object of a function that is registered or saved in the project by using `project.get_function(key)`.

Example:

```python
    # get the data-prep function, add volume mount and run it with data input
    project.get_function("data-prep").apply(v3io_mount())
    run = project.run_function("data-prep", inputs={"data": data_url})
```

(Run_project_functions)=
## Run, build, and deploy functions

There is a set of methods used to deploy and run project functions. They can be used interactively or inside a pipeline 
(inside a pipeline it will be automatically mapped to the relevant pipeline engine command).

* {py:meth}`~mlrun.projects.run_function`  - Run a local or remote task as part of a local run or pipeline
* {py:meth}`~mlrun.projects.build_function`  - deploy an ML function, build a container with its dependencies for use in runs
* {py:meth}`~mlrun.projects.deploy_function`  - deploy real-time/online (nuclio or serving based) functions

You can use those methods as `project` methods, or as global (`mlrun.`) methods. The current project is assumed for the later case.

    run = myproject.run_function("train", inputs={"data": data_url})  # will run the "train" function in myproject
    run = mlrun.run_function("train", inputs={"data": data_url})  # will run the "train" function in the current/active project
    
The first parameter in those three methods is the function name (in the project), or it can be a function object if you want to use functions you imported/created ad hoc, for example:

    # import a serving function from the marketplace and deploy a trained model over it
    serving = import_function("hub://v2_model_server", new_name="serving")
    deploy = deploy_function(
        serving,
        models=[{"key": "mymodel", "model_path": train.outputs["model"]}],
    )
