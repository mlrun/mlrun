<a id="top"></a>
# Projects <!-- omit in toc -->

A Project is a container for all your work on a particular activity. All the associated code, jobs and artifacts are organized within the projects. A project is also a great way to collaborate with others, since you can share your work, as well as create projects based on existing projects.

- [Creating a new project](#creating-a-new-project)
- [Setting up Git Remote Repository](#setting-up-git-remote-repository)
- [Loading existing projects](#loading-existing-projects)
- [Updating the project and code](#updating-the-project-and-code)

## Creating a new project

It's a best practice to have all your notebooks associated with a project. An easy way to do that is to create a project in the beginning of the notebook using the `new_project` MLRun method, which receives the following parameters:

- **`name`** (Required) &mdash; the project name.
- **`context`** &mdash; the path to a local project directory (the project's context directory).
  The project directory contains a project-configuration file (default: **project.yaml**), which defines the project, and additional generated Python code.
  The project file is created when you save your project (using the `save` MLRun project method), as demonstrated in Step 6.
- **`functions`** &mdash; a list of functions objects or links to function code or objects.
- **`init_git`** &mdash; set to `True` to perform Git initialization of the project directory (`context`).
  > **Note:** It's customary to store project code and definitions in a Git repository.

Projects are visible in the MLRun dashboard only after they're saved to the MLRun database, which happens whenever you run code for a project.

For example, use the following code to create a project named **my-project** and stores the project definition in a subfolder named `conf`:

```python
from os import path
from mlrun import new_project

project_name = 'my-project'
project_path = path.abspath('conf')
project = new_project(project_name, project_path, init_git=True)

print(f'Project path: {project_path}\nProject name: {project_name}')
```

You can also ensure the project name is unique, by concatenating your current username. For example, the following code concatenates the **V3IO_USERNAME** environment variable to the project name:

```python
from os import getenv
project_name = '-'.join(filter(None, ['my-project', getenv('V3IO_USERNAME', None)]))
```

## Setting up Git Remote Repository
It is also highly recommended to set up a remote repository in order to save your work on git. To do that, you need to call `create_remote`. For example, to set up a remote repository on GitHub:

``` python
remote_git = 'https://github.com/<my-org>/<my-repo>.git'
project.create_remote(remote_git)
```

In case the remote repository has existing content, you should pull from it: 
``` python
project.pull()
```

## Loading existing projects

You can use an existing project as a baseline by calling the `load_project` function. This enables reuse of existing code and definitions.

Projects can be stored in a Git repo or in a tar archive (on object store like S3, V3IO).

For example, to load the **demo-xgb-project** to `my_proj` the user's home directory:

``` python
from os import path
from pathlib import Path
# source Git Repo
# YOU SHOULD fork this to your account and use the fork if you plan on modifying the code
url = 'git://github.com/mlrun/demo-xgb-project.git' # refs/tags/v0.4.7'

# alternatively can use tar files, e.g.
#url = 'v3io:///users/admin/tars/src-project.tar.gz'

# change if you want to clone into a different dir, can use clone=True to override the dir content
project_dir = path.join(str(Path.home()), 'my_proj')
project = load_project(project_dir, url, clone=True)

```

> **Note**: If URL is not specified it will use the context and search for Git repo inside it, or use the init_git=True flag to initialize a Git repo in the target context directory.

## Updating the project and code

A user can update the code using the standard Git process (commit, push, ..), if you update/edit the project object you need to run `proj.save()` which will update the `project.yaml` file in your context directory, followed by pushing your updates.

You can use `proj.push(branch, commit_message, add=[])` which will do the work for you (save the yaml, commit updates, push)

> Note: you must push updates before you build functions or run workflows since the builder will pull the code from the git repo.

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

If you want to pull changes done by others use `proj.pull()`, if you need to update the project spec (since the yaml file changed) use `proj.reload()` and for updating the local/remote function specs use `proj.sync_functions()` (or add `sync=True` to `.reload()`).

``` python
project.pull()
```


[Back to top](#top)
