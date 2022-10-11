(create-projects)=
# Create and load projects

Projects refer to a `context` directory that holds all the project code and configuration. The `context` dir is 
usually mapped to a `git` repository and/or to an IDE (PyCharm, VSCode, etc.) project.   

There are three ways to create/load a `project` object:
* {py:meth}`~mlrun.projects.new_project`  &mdash; Create a new MLRun project and optionally load it from a yaml/zip/git template.
* {py:meth}`~mlrun.projects.load_project` &mdash; Load a project from a context directory or remote git/zip/tar archive.
* {py:meth}`~mlrun.projects.get_or_create_project` &mdash; Load a project from the MLRun DB if it exists, or from a specified 
  context/archive. 

Projects can also be loaded and workflows/pipelines can be executed using the CLI (using the `mlrun project` command).

```{admonition} Note
Data-access permissions are given to the original creator of files. If you transfer ownership on a project to a user in a different user group, then you must give the new owner the relevant permissions on the data files and folders of the project (by modifying the POSIX permissions in the file-system on the project files if possible). Otherwise, the user will not be able to work with the project data.
```

**In this section**
- [Creating a new project](#creating-a-new-project)
- [Load and run projects from context, git or archive](#load-and-run-projects-from-context-git-or-archive)
- [Get a project from DB or create it (get_or_create_project)](#get-from-db-or-create-get-or-create-project)

## Creating a new project

To define a new project from scratch, use {py:meth}`~mlrun.projects.new_project`. You must specify a `name`, 
location for the `context` directory (e.g. `./`) and other optional parameters (see below).
The `context` dir holds the configuration, code, and workflow files. File paths in the project are relative to the context root.

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


When projects are saved a `project.yaml` file with project definitions is written to the `context` dir. Alternatively, you
can manually create the `project.yaml` file and load it using `load_project()` or the `from_template` parameter.
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
    # add another marketplace function and save
    project.set_function('hub://test_classifier', 'test')  
    project.save()      
```

```{admonition} Note
* Projects are visible in the MLRun dashboard only after they're saved to the MLRun database (with `.save()`) or after the workflows are executed (with `.run()`).
* You can ensure the project name is unique per user by setting the `user_project` parameter to `True`.
```

## Load and run projects from context, git or archive

When a project is already created and stored in a git archive you can quickly load and use it with the 
{py:meth}`~mlrun.projects.load_project` method. `load_project` uses a local context directory (with initialized `git`) 
or clones a remote repo into the local dir and returns a project object.

You need to provide the path to the `context` dir and the git/zip/tar archive `url`. The `name` can be specified or taken 
from the project object, they can also specify `secrets` (repo credentials), `init_git` flag (to initialize git in the context dir), 
`clone` flag (indicating we must clone and ignore/remove local copy), and `user_project` flag (indicate the project name is unique to the user).

Example of loading a project from git and running the `main` workflow:

```python
    project = mlrun.load_project("./", "git://github.com/mlrun/project-demo.git")
    project.run("main", arguments={'data': data_url})
```

```{admonition} Note
If the `url` parameter is not specified it searches for Git repo inside the context dir and uses its metadata, 
or uses the init_git=True flag to initialize a Git repo in the target context directory.
```

### Load and run using the CLI

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

## Get a project from DB or create it (`get_or_create_project`)

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

