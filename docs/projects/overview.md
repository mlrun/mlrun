# Projects, Automation & CI/CD

A Project is a container for all your work on a particular activity. All the associated code, functions, 
jobs/workflows and artifacts are organized within the projects. Projects can be mapped to `GIT` repositories which 
enable versioning, collaboration, and CI/CD.

Users can create project definitions using the SDK or a Yaml file and store those in MLRun DB, file, or archive.
Project definitions include lists of parameters, functions, workflows (pipelines), and artifacts. 
Once the project is loaded you can run jobs/workflows which refer to any project element by name, 
allowing separation between configuration and code.

Projects refer to a `context` directory which holds all the project code and configuration, the `context` dir is 
usually mapped to a `git` repository and/or to an IDE (PyCharm, VSCode, ..) project.   

There are three ways to create/load a `project` object:
* {py:meth}`~mlrun.projects.new_project`  - Create a new MLRun project and optionally load it from a yaml/zip/git template
* {py:meth}`~mlrun.projects.load_project` - Load a project from a context directory or remote git/zip/tar archive 
* {py:meth}`~mlrun.projects.get_or_create_project` - Load a project from the MLRun DB if it exists, or from specified 
  context/archive. 

once we create a project we can add/update its functions, artifacts, or workflows using {py:meth}`~mlrun.projects.MlrunProject.set_function`, 
{py:meth}`~mlrun.projects.MlrunProject.set_artifact`, {py:meth}`~mlrun.projects.MlrunProject.set_workflow`, and set
various project attributes (`parameters`, `secrets`, etc.)

we use the project {py:meth}`~mlrun.projects.MlrunProject.run` method to run a registered workflow using a pipeline engine (e.g. Kubeflow pipelines),
workflow execute the registered functions in a sequence/graph (DAG), can reference project parameters, secrets and artifacts by name.

Projects can also be loaded and workflows/pipelines can be executed using the CLI (using `mlrun project` command)

- [Creating a new project](#creating-a-new-project)
- [Load & Run projects from context, git or archive](#setting-up-git-remote-repository)
- [Get from DB or create (get_or_create_project)](#get-from-db-or-create-get-or-create-project)
- [Working with Git](#working-with-git)
- [Updating and using project functions](#updating-and-using-project-functions)
- [Using workflows for project automation and CI/CD](#using-workflows-for-project-automation-and-ci-cd)

## Creating a new project

To define a new project from scratch we use {py:meth}`~mlrun.projects.new_project`, we must specify a `name`, 
location for the `context` directory (e.g. `./`) and other optional parameters (see below).
The `context` dir holds the configuration, code, and workflow files, file paths in the project are relative to the context root.

```python
    # create a project with local and marketplace functions
    project = mlrun.new_project("myproj", "./", init_git=True, description="my new project")
    project.set_function('prep_data.py', 'prep-data', image='mlrun/mlrun', handler='prep_data')
    project.set_function('hub://sklearn_classifier', 'train')
    
    # register a simple named artifact in the project (to be used in workflows)  
    data_url = 'https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv'
    project.set_workflow('main', "./myflow.py")

    # add a multi-stage workflow (./myflow.py) to the project with the name 'main' and save the project 
    project.set_artifact('data', Artifact(target_path=data_url))
    project.save()

    # run the "main" workflow (watch=True to wait for run completion)
    project.run("main", watch=True)
```


When projects are saved a `project.yaml` file with project definitions is written to the context dir (alternatively we
can manually create the `project.yaml` file and load it using `load_project()` or the `from_template` parameter).
the generated `project.yaml` for the above project will look like:

```yaml
kind: project
metadata:
  name: myproj
spec:
  description: my new project
  functions:
  - url: prep_data.py
    name: prep-data
    image: mlrun/mlrun
    handler: prep_data
  - url: hub://sklearn_classifier
    name: train
  workflows:
  - name: main
    path: ./myflow.py
    engine: kfp
  artifacts:
  - kind: ''
    target_path: https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv
    key: data
```
 
Projects can also be created from a template (yaml file, zip file, or git repo), allowing users to create reusable skeletons,
content of the zip/tar/git archive is copied into the context dir.

The `init_git` flag is used to initialize git in the context dir, `remote` attribute is used to register the remote 
git repository URL, and `user_project` flag indicate the project name is unique to the user. 

Example create a new project from a zip template:

```python
    # create a project from zip, initialize a local git, and register the git remote path
    project = mlrun.new_project("myproj", "./", init_git=True, user_project=True,
                                remote="git://github.com/mlrun/demo-xgb-project.git",
                                from_template="http://mysite/proj.zip")
    # add another marketplace function and save
    project.set_function('hub://test_classifier', 'test')  
    project.save()      
```

```{admonition} Note
* Projects are visible in the MLRun dashboard only after they're saved to the MLRun database (with `.save()`) or workflows are executed (with `.run()`).
* You can ensure the project name is unique per user by setting the the `user_project` parameter to `True`
```

## Load & Run projects from context, git or archive

When our project is already created and stored in a git archive we can quickly load and use it with the 
{py:meth}`~mlrun.projects.load_project` method. `load_project` will use a local context directory (with initialized `git`) 
or clone a remote repo into the local dir and return a project object.

Users need to provide the path to the `context` dir and the git/zip/tar archive `url`, the `name` can be specified or taken 
from the project object, they can also specify `secrets` (repo credentials), `init_git` flag (to initialize git in the context dir), 
`clone` flag (indicating we must clone and ignore/remove local copy), and `user_project` flag (indicate the project name is unique to the user).

example, load a project from git and run the `main` workflow:

```python
    project = mlrun.load_project("./", "git://github.com/mlrun/project-demo.git")
    project.run("main", arguments={'data': data_url})
```

```{admonition} Note
If the `url` parameter is not specified it will search for Git repo inside the context dir and use its metadata, 
or use the init_git=True flag to initialize a Git repo in the target context directory.
```

### Load & run using the CLI

Loading a project from `git` into `./` :

```
mlrun project -n myproj -u "git://github.com/mlrun/project-demo.git" .
```

Running a specific workflow (`main`) from the project stored in `.` (current dir):

```
mlrun project -r main -w .
```

**CLI usage details:**

```
Usage: mlrun project [OPTIONS] [CONTEXT]

Options:
  -n, --name TEXT           project name
  -u, --url TEXT            remote git or archive url
  -r, --run TEXT            run workflow name of .py file
  -a, --arguments TEXT      pipeline arguments name and value tuples (with -r flag),
                            e.g. -a x=6

  -p, --artifact-path TEXT  output artifacts path if not default
  -x, --param TEXT          mlrun project parameter name and value tuples,
                            e.g. -p x=37 -p y='text'

  -s, --secrets TEXT        secrets file=<filename> or env=ENV_KEY1,..
  --init-git                for new projects init git context
  -c, --clone               force override/clone into the context dir
  --sync                    sync functions into db
  -w, --watch               wait for pipeline completion (with -r flag)
  -d, --dirty               allow run with uncommitted git changes
```

## Get from DB or create (`get_or_create_project`)

If you already have a project saved in the DB and you need to access/use it (for example from a different notebook or file), 
use the {py:meth}`~mlrun.projects.get_or_create_project` method. It will first try to read the project from the DB, 
and only if it doesnt exist in the DB it will load/create it. 

```{admonition} Note
If you update the project object from different files/notebooks/users, make sure you `.save()` your project after a change, 
and run `get_or_create_project` to load changes made by others. 
```

Example:

```python
    # load project from the DB (if exist) or the source repo
    project = mlrun.get_or_create_project("myproj", "./", "git://github.com/mlrun/demo-xgb-project.git")
    project.pull("development")  # pull the latest code from git
    project.run("main", arguments={'data': data_url})  # run the workflow "main"
```


## Working with Git

A user can update the code using the standard Git process (commit, push, ..), if you update/edit the project object you 
need to run `project.save()` which will update the `project.yaml` file in your context directory, followed by pushing your updates.

You can use the standard `git` cli to `pull`, `commit`, `push`, etc. MLRun project will sync with the local git state.
You can also use project methods with the same functionality, it simplifies the work for common task but does not expose the full git functionality.

* **{py:meth}`~mlrun.projects.MlrunProject.pull`** - pull/update sources from git or tar into the context dir
* **{py:meth}`~mlrun.projects.MlrunProject.create_remote`** - create remote for the project git
* **{py:meth}`~mlrun.projects.MlrunProject.push`** - save project state and commit/push updates to remote git repo

e.g. `proj.push(branch, commit_message, add=[])` will save the state to DB & yaml, commit updates, push

```{admonition} Note
you must push updates before you build functions or run workflows which use code from git,
since the builder or containers will pull the code from the git repo.
```

If you are using containerized Jupyter you may need to first set your Git parameters, e.g. using the following commands:

```
git config --global user.email "<my@email.com>"
git config --global user.name "<name>"
git config --global credential.helper store
```

After that you would need to login once to git with your password as well as restart the notebook

``` python
project.push('master', 'some edits')
```

## Updating and using project functions

Projects host or link to functions which are used in job or workflow runs. you add functions to a project using 
{py:meth}`~mlrun.projects.MlrunProject.set_function` this will register them as part of the project definition (and Yaml file),
alternatively you can create functions using methods like {py:func}`~mlrun.run.code_to_function` and save them to the DB (under the same project). 
the preferred approach would be to use `set_function` (which also records the functions in the project spec).

The {py:meth}`~mlrun.projects.MlrunProject.set_function` method allow you to add/update many types of functions:
* **marketplace functions** - load/register a marketplace function into the project (func="hub://...")
* **notebook file** - convert a notebook file into a function (func="path/to/file.ipynb")
* **python file** - convert a python file into a function (func="path/to/file.py")
* **database function** - function stored in MLRun DB (func="db://project/func-name:version")
* **function yaml file** - read the function object from a yaml file (func="path/to/file.yaml")
* **inline function spec** - save the full function spec in the project definition file (func=func_object), not recommended

When loading a function from code file (py, ipynb) you should also specify a container `image` and the runtime `kind` (will use `job` kind as default),
you can optionally specify the function `handler` (the function handler to invoke), and a `name`.

If the function is not a single file function, and it require access to multiple files/libraries in the project, 
you should set the `with_repo=True` which will add the entire repo code into the destination container during build or run time.

```{admonition} Note
when using `with_repo=True` the functions need to be deployed (`function.deploy()`) to build a container, unless you set `project.spec.load_source_on_run=True` which instructs MLRun to load the git/archive repo into the function container 
at run time and do not require a build (this is simpler when developing, for production its preferred to build the image with the code)
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

once functions are registered or saved in the project we can get their function object using `project.func(key)`.

example:

```python
    # get the data-prep function, add volume mount and run it with data input
    run = project.func("data-prep").apply(v3io_mount()).run(inputs={"data": data_url})
```

When running inside a workflow the `funcs` dictionary holds the function object (enriched with workflow metadata/spec elements)

## Using workflows for project automation and CI/CD

Workflows are used to run multiple dependent steps in a graph (DAG) which execute project functions and access project data, parameters, secrets. 

Example workflow:

```python
from kfp import dsl
import mlrun

funcs = {}
project = mlrun.projects.pipeline_context.project
default_pkg_class = "sklearn.linear_model.LogisticRegression"

@dsl.pipeline(name="Demo training pipeline", description="Shows how to use mlrun.")
def kfpipeline(model_pkg_class=default_pkg_class, build=0):

    # if build=1, build the function image before the run
    with dsl.Condition(build == 1) as build_cond:
        funcs["prep-data"].deploy_step()

    # run a local data prep function
    prep_data = funcs["prep-data"].as_step(
        name="prep_data",
        inputs={"source_url": project.get_artifact_uri("data")},
        outputs=["cleaned_data"],
    ).after(build_cond)

    # train the model using a library (hub://) function and the generated data
    train = funcs["train"].as_step(
        name="train",
        inputs={"dataset": prep_data.outputs["cleaned_data"]},
        params={
            "model_pkg_class": model_pkg_class,
            "label_column": project.get_param("label", "label"),
        },
        outputs=["model", "test_set"],
    )

    # test the model using a library (hub://) function and the generated model
    funcs["test"].as_step(
        name="test",
        params={"label_column": "label"},
        inputs={
            "models_path": train.outputs["model"],
            "test_set": train.outputs["test_set"],
        },
    )
```
