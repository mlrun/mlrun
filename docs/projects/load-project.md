(load-project)=
# Load and run projects

Project code, metadata, and configuration are stored and versioned in source control systems like GIT or archives (zip, tar) 
and can be loaded into your work environment or CI system with a single SDK or CLI command.

<p align="center"><img src="../_static/images/project-lifecycle.png" alt="project-lifecycle" width="700"/></p><br>

The project root (context) directory contains the `project.yaml` file with required metadata and links to various project files/objects, and is read during the `load` process.

**In this section**
- [Load projects using the SDK](#load-sdk)
- [Load projects using the CLI](#load-cli)

See also details on loading and using projects [**with CI/CD frameworks**](./ci-integration.html).

<a id='load-sdk'></a>
## Load projects using the SDK

When a project is already created and stored in a local dir, git. or archive you can quickly load and use it with the 
{py:meth}`~mlrun.projects.load_project` method. `load_project` uses a local context directory (with initialized `git`) 
or clones a remote repo into the local dir and returns a project object.

You need to provide the path to the `context` dir and the git/zip/tar archive `url`. The `name` can be specified or taken 
from the project object, they can also specify `secrets` (dict with repo credentials), `init_git` flag (to initialize git in the context dir), 
`clone` flag (indicating we must clone and ignore/remove local copy), and `user_project` flag (indicate the project name is unique to the user).

Example of loading a project from git and running the `main` workflow:

```python
# load the project and run the 'main' workflow
project = load_project(context="./", name="myproj", url="git://github.com/mlrun/project-archive.git")
project.run("main", arguments={'data': data_url})
```

```{admonition} Note
If the `url` parameter is not specified it searches for Git repo inside the context dir and uses its metadata, 
or uses the init_git=True flag to initialize a Git repo in the target context directory.
```

Onc
e the project object is loaded use the {py:meth}`~mlrun.projects.MlrunProject.run` method to execute workflows, see details on [**building and running workflows**](./build-run-workflows-pipelines.html)), 
and how to [**run, build, or deploy**](./run-build-deploy.html) individual functions. 

You can edit or add project elements like functions, workflows, artifacts, etc. (see:  [**create and use projects**](./create-project.html)).
Once you make changes use GIT or MLRun commands to push those changes to the archive (see: [**save into git or an archive**](./create-project.html#push)).

<a id='load-cli'></a>
## Load projects using the CLI

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

