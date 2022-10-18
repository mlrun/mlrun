(create-projects)=
# Create, save and use projects

A project is a container for all the assets, configuration, and code of a particular application. It is the starting point for your work. Projects are stored in a versioned source repository (GIT) or archive and can map to IDE projects (in PyCharm, VSCode, etc.).

<p align="center"><img src="../_static/images/project.png" alt="mlrun-project" width="600"/></p><br>

**In this section**
- [Creating a new project](#create)
- [Adding and updating project elements](#add-elements)
- [Pushing the project content into git or an archive](#push)
- [Running functions and workflows](#run)
- [Get a project from DB or create it](#get-or-ceate)

<a id="create"></a>
## Creating a new project

Project files (code, configuration, etc.) are stored in a directory (the project `context` path) and can be pushed to or loaded from the source repository. See the following project directory example:

```
my-project           # Parent directory of the project (context)
├── data             # Project data for local tests or outputs (not tracked by version control)
├── docs             # Project Documentation
├── notebooks        # Project related Jupyter notebooks (can be used for tests, experiments, visualization)
├── src              # Project source code (functions, libs, workflows)
├── tests            # Unit tests (pytest) for the different functions
├── project.yaml     # MLRun Project spec file
├── README.md        # Project README
└── requirements.txt # Default Python requierments file (can have function specific requierments as well)
```


To define a new project from scratch, use {py:meth}`~mlrun.projects.new_project`. You must specify a `name`, 
location for the `context` directory (e.g. `./`) and other optional parameters (see below).
The `context` dir holds the configuration, code, and workflow files. File paths in the project are relative to the context root.

```python
    # create a project with local and marketplace functions
    project = mlrun.new_project("myproj", "./", init_git=True, description="my new project")
```
 
Projects can also be created from a template (yaml file, zip file, or git repo), allowing users to create reusable skeletons. The
content of the zip/tar/git archive is copied into the context dir.

The `init_git` flag is used to initialize git in the context dir, the `remote` attribute is used to register the remote 
git repository URL, and the `user_project` flag indicates that the project name is unique to the user. 

Example of creating a new project from a zip template:

```python
    # create a project from zip, initialize a local git, and register the git remote path
    project = mlrun.new_project("myproj", "./", init_git=True, user_project=True,
                                remote="git://github.com/mlrun/demo-xgb-project.git",
                                from_template="http://mysite/proj.zip")
```

<a id="add-elements"></a>
## Adding and updating project elements

Projects host [functions](../runtimes/functions.html), [workflows](../concepts/workflow-overview.html), [artifacts (datasets, models, etc.)](../store/artifacts.html), [features (sets, vectors)](../feature-store/feature-store.html), and configuration (parameters, [secrets](../secrets.html), source, etc.).

**Adding functions:**

Function with basic attributes such as code, requirements, image, etc. can be registered using the `set_function()` command.
Functions can be created from a single code/notebook file or have access to the entire project context directory (by adding the `with_repo=True` flag, it will guarantee the project context is cloned into the function runtime environment).


```python
    # register a (single) python file as a function
    project.set_function('src/prep_data.py', 'prep-data', image='mlrun/mlrun', handler='prep')

    # register a notebook file as a function, specify custom image and extra requirements 
    project.set_function('mynb.ipynb', name='test-function', 
                         image="my-org/my-image", handler="run_test", requirements="requirements.txt")

    # register a module.handler as a function (require defining the source/working dir)
    project.spec.workdir = "src"
    project.set_function(name="train", handler="training.train",  image="mlrun/mlrun", kind="job", with_repo=True)
```

See [**details and examples**]() for how to create and register different kinds of functions

**Register additional project objects and metadata:**

You can define other objects (workflows, artifacts, secrets) and parameters in the project and use them in your functions, for example:

```python
    # register a simple named artifact in the project (to be used in workflows)  
    data_url = 'https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv'
    project.set_artifact('data', target_path=data_url)

    # add a multi-stage workflow (./myflow.py) to the project with the name 'main' and save the project 
    project.set_workflow('main', "./myflow.py")
    
    # read env vars from dict or file and set as project secrets
    project.set_secrets({"SECRET1": "value"})
    project.set_secrets(file_path="secrets.env")
    
    project.spec.params = {"x": 5}
```

**Save the project:**

```python
    # save the project in the db (and into the project.yaml file)
    project.save()
```

````{dropdown} show the generated project.yaml file
The generated `project.yaml` for the above project looks like:

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
````

```{admonition} Note
* Projects are visible in the MLRun dashboard only after they're saved to the MLRun database (with `.save()`) or after the workflows are executed (with `.run()`).
* You can ensure the project name is unique per user by setting the `user_project` parameter to `True`.
```

<a id="push"></a>
## Pushing the project content into git or an archive

Use standard Git commands to push the current project tree into a git archive, make sure you `.save()` the project before pushing it

    git remote add origin <server>
    git commit -m "Commit message"
    git push origin master

Alternatively you can use MLRun SDK calls:
- `project.create_remote(git_uri, branch=branch)` - to register the remote Git path
- `project.push()` - save project spec (`project.yaml`) and commit/push updates to remote repo

````{admonition} Note
If you are using containerized Jupyter you might need to first set your Git parameters, e.g. using the following commands:

```
git config --global user.email "<my@email.com>"
git config --global user.name "<name>"
git config --global credential.helper store
```

````

you can also save the project content and metadata into a local or remote `.zip` archive, examples: 

    project.export("../archive1.zip")
    project.export("s3://my-bucket/archive1.zip")
    project.export(f"v3io://projects/{project.name}/archive1.zip")

<a id="run"></a>
## Running functions and workflows 

```{admonition} Note
You must push updates before you build functions or run workflows which use code from git,
since the builder or containers pull the code from the git repo.
```


```python
    # run the "main" workflow (watch=True to wait for run completion)
    project.run("main", watch=True)
```

<a id="get-or-create"></a>
## Get a project from DB or create it

If you already have a project saved in the DB and you need to access/use it (for example from a different notebook or file), 
use the {py:meth}`~mlrun.projects.get_or_create_project` method. It first tries to read the project from the DB, 
and only if it doesn't exist in the DB it loads/creates it. 

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

