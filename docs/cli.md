(cli)=
# Command-Line Interface

- [CLI commands](#cli-commands)
- [Building and running a function from a Git Repository](#building-and-running-a-function-from-a-git-repository)
- [Using a sources archive](#using-a-sources-archive)

<a id="cli-commands"></a>
## CLI commands

Use the following commands of the MLRun command-line interface (CLI) &mdash; `mlrun` &mdash; to build and run MLRun functions:

- [`build`](#cli-cmd-build)
- [`clean`](#cli-cmd-clean)
- [`config`](#cli-cmd-config)
- [`get`](#cli-cmd-get)
- [`logs`](#cli-cmd-logs)
- [`project`](#cli-cmd-project)
- [`run`](#cli-cmd-run)
- [`version`](#cli-cmd-version)
- [`watch-stream`](#cli-cmd-watch-stream)

Each command supports many flags, some of which are listed in their relevant sections. To view all the flags of a command, run `mlrun <command name> --help`.

<a id="cli-cmd-build"></a>
### `build` 

Use the `build` CLI command to build all the function dependencies from the function specification into a function container (Docker image).

Usage: mlrun build [OPTIONS] FUNC_URL

Example:  `mlrun build myfunc.yaml`

| Flag                               | Description                                                                                                                                                                                        |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| &minus;&minus;name                 | Function name                                                                                                                                                                                      |
| &minus;&minus;project              | Project name                                                                                                                                                                                       |
| &minus;&minus;tag                  | Function tag                                                                                                                                                                                       |
| -i, &minus;&minus;image            | Target image path                                                                                                                                                                                  |
| -s, &minus;&minus;source           | Path/URL of the function source code. A PY file, or if `-a/--archive`  is specified, a directory to archive. (Default: './')                                                                       |
| -b, &minus;&minus;base-image       | Base Docker image                                                                                                                                                                                  |
| -c, &minus;&minus;command          | Build commands; for example, '-c pip install pandas'                                                                                                                                               |
| &minus;&minus;secret&minus;name    | Name of a container-registry secret                                                                                                                                                                |
| -a, &minus;&minus;archive          | Path/URL of a target function-sources archive directory: as part of the build, the function sources (see `-s/--source`) are archived into a TAR file and then extracted into the archive directory |
| &minus;&minus;silent               | Do not show build logs                                                                                                                                                                             |
| &minus;&minus;with&minus;mlrun     | Add the MLRun package ("mlrun")                                                                                                                                                                    |
| &minus;&minus;db                   | Save the run results to path or DB url                                                                                                                                                             |
| -r, &minus;&minus;runtime          | Function spec dict, for pipeline usage                                                                                                                                                             |
| &minus;&minus;kfp                  | Running inside Kubeflow Piplines, do not use                                                                                                                                                       |
| &minus;&minus;skip                 | Skip if already deployed                                                                                                                                                                           |
| &minus;&minus;env&minus;file       | Path to .env file to load config/variables from                                                                                                                                                    |
| &minus;&minus;ensure&minus;project | Ensure the project exists, if not, create project                                                                                                                                                  |


> **Note:** For information about using the `-a|--archive` option to create a function-sources archive, see [Using a Sources Archive](#sources-archive) later in this tutorial.

<a id="cli-cmd-clean"></a>
### `clean` 

Use the `clean` CLI command to clean runtime resources. When run without any flags, it cleans the resources for all runs of all runtimes.

Usage: mlrun clean [OPTIONS] [KIND] [id]

Examples: 

- Clean resources for all runs of all runtimes:  `mlrun clean`
- Clean resources for all runs of a specific kind (e.g. job):  `mlrun clean job`
- Clean resources for specific job (by uid):  `mlrun clean mpijob 15d04c19c2194c0a8efb26ea3017254b`


| Flag                                                         | Description                                                                                                                                                                                                                                                          |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| &minus;&minus;kind &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Clean resources for all runs of a specific kind (e.g. job).                                                                                                                                                                                                          |
| &minus;&minus;id &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   | Delete the resources of the mlrun object twith this identifier. For most function runtimes, runtime resources are per Run, and the identifier is the Run’s UID. For DASK runtime, the runtime resources are per Function, and the identifier is the Function’s name. |

  
| Options                                       | Description                                                                                                                                         |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| &minus;&minus;api                             | URL of the mlrun-api service.                                                                                                                       |
| &minus;ls, &minus;&minus;label&minus;selector | Delete only runtime resources matching the label selector.                                                                                          |
| &minus;f, &minus;&minus;force                 | Delete the runtime resource even if they're not in terminal state or if the grace period didn’t pass.                                               |
| &minus;gp, &minus;&minus;grace&minus;period   | Grace period, in seconds, given to the runtime resource before they are actually removed, counted from the moment they moved to the terminal state. |

<a id="cli-cmd-config"></a>
### `config` 

Use the `config` CLI command to show the mlrun client environment configuration, such as location of artifacts and api.

Example:  `mlrun config`

| Flag                                  | Description                                              |
|---------------------------------------|----------------------------------------------------------|
| &minus;&minus;command                 | get (default), set or clear                              |
| &minus;&minus;env&minus;file          | Path to the mlrun .env file (defaults to '~/.mlrun.env') |
| -a, &minus;&minus;api                 | API service url                                          |
| -p, &minus;&minus;artifact&minus;path | Default artifacts path                                   |
| -u, &minus;&minus;username            | Username (for remote access)                             |
| -k, &minus;&minus;access-key          | Access key (for remote access)                           |
| -e, &minus;&minus;env&minus;vars      | Additional env vars, e.g. -e AWS_ACCESS_KEY_ID=<key-id>  |


<a id="cli-cmd-get"></a>
### `get` 

Use the `get` CLI command to list one or more objects per kind/class.

Usage: get pods | runs | artifacts | func [name]

Examples:

- `mlrun get runs --project getting-started-admin`
- `mlrun get pods --project getting-started-admin`
- `mlrun get artifacts --project getting-started-admin`
- `mlrun get func prep-data --project getting-started-admin`

| Flag                        | Description                                                           |
|-----------------------------|-----------------------------------------------------------------------|
| &minus;&minus;kind          | resource type to list/get: run, runtime, workflow, artifact, function |
| &minus;&minus;name          | Name of object to return                                              |
| -s, &minus;&minus;selector  | Label selector                                                        |
| -n, &minus;&minus;namespace | Kubernetes namespace                                                  |
| &minus;&minus;uid           | Object ID                                                             |
| &minus;&minus;project       | Project name to return                                                |
| -t, &minus;&minus;tag       | Artifact/function tag of object to return                             |
| &minus;&minus;db            | db path/url of object to return                                       |


<a id="cli-cmd-logs"></a>
### `logs` 

Use the `logs` CLI command to get or watch task logs.

Usage: logs [OPTIONS] uid

Example:  `mlrun logs ba409c0cb4904d60aa8f8d1c05b40a75 --project getting-started-admin`

| Flag                      | Description                                                                                                                                                                                                                                                                   |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| &minus;&minus;uid         | UID of the run to get logs for                                                                                                                                                                                                                                                |
| -p, &minus;&minus;project | Project name                                                                                                                                                                                                                                                                  |
| &minus;&minus;offset      | Retrieve partial log, get up to size bytes starting at the offset from beginning of log                                                                                                                                                                                       |
| &minus;&minus;db          | API service url                                                                                                                                                                                                                                                               |
| -w, &minus;&minus;watch   | Retrieve logs of a running process, and watch the progress of the execution until it completes. Prints out the logs and continues to periodically poll for, and print, new logs as long as the state of the runtime that generates this log is either `pending` or `running`. |


<a id="cli-cmd-project"></a>
### `project` 
    
Use the `project` CLI command to load and/or run a project.

Usage: mlrun project [OPTIONS] [CONTEXT]

Example:  `mlrun project -r workflow.py .`


| Flag                                           | Description                                                                                                                                                                                                                                                        |
|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|     
| -n, &minus;&minus;context                      | Project context path                                                                                                                                                                                                                                               |
| -n, &minus;&minus;name                         | Project name                                                                                                                                                                                                                                                       |
| -u, &minus;&minus;url                          | Remote git or archive url of the project                                                                                                                                                                                                                           |
| -r, &minus;&minus;run                          | Run workflow name of .py file                                                                                                                                                                                                                                      |
| -a, &minus;&minus;arguments                    | Kubeflow pipeline arguments name and value tuples (with -r flag), e.g. -a x=6                                                                                                                                                                                      |
| -p, &minus;&minus;artifact&minus;path          | Target path/url for workflow artifacts.  The string `{{workflow.uid}}` is replaced by workflow id                                                                                                                                                                  | 
| -x, &minus;&minus;param                        | mlrun project parameter name and value tuples, e.g. -p x=37 -p y='text'                                                                                                                                                                                            |
| -s, &minus;&minus;secrets                      | Secrets file=<filename> or env=ENV_KEY1,..                                                                                                                                                                                                                         |
| &minus;&minus;namespace                        | k8s namespace                                                                                                                                                                                                                                                      |
| &minus;&minus;db                               | API service url                                                                                                                                                                                                                                                    |
| &minus;&minus;init&minus;git                   | For new projects init git the context dir                                                                                                                                                                                                                          |
| -c, &minus;&minus;clone                        | Force override/clone into the context dir                                                                                                                                                                                                                          |
| &minus;&minus;sync                             | Sync functions into db                                                                                                                                                                                                                                             |
| -w, &minus;&minus;watch                        | Wait for pipeline completion (with -r flag)                                                                                                                                                                                                                        |
| -d, &minus;&minus;dirty                        | Allow run with uncommitted git changes                                                                                                                                                                                                                             |
| &minus;&minus;git&minus;repo                   | git repo (org/repo) for git comments                                                                                                                                                                                                                               |
| &minus;&minus;git&minus;issue                  | git issue number for git comments                                                                                                                                                                                                                                  |
| &minus;&minus;handler                          | Workflow function handler name                                                                                                                                                                                                                                     |
| &minus;&minus;engine                           | Workflow engine (kfp/local)                                                                                                                                                                                                                                        |
| &minus;&minus;local                            | Try to run workflow functions locally                                                                                                                                                                                                                              |
| &minus;&minus;timeout                          | Timeout in seconds to wait for pipeline completion (used when watch=True)                                                                                                                                                                                          |
| &minus;&minus;env&minus;file                   | Path to .env file to load config/variables from                                                                                                                                                                                                                    |
| &minus;&minus;save/&minus;&minus;no&minus;save | Create and save the project if not exist (default to save)                                                                                                                                                                                                         |
| &minus;&minus;schedule                         | To create a schedule, define a standard crontab expression string. To use the pre-defined workflow's schedule: `set --schedule 'true'`. [See cron details](https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html#module-apscheduler.triggers.cron). |
| &minus;&minus;save&minus;secrets               | Store the project secrets as k8s secrets                                                                                                                                                                                                                           |
| -nt, &minus;&minus;notifications               | To have a notification for the run set notification file destination define: file=notification.json or a 'dictionary configuration e.g \'{"slack":{"webhook":"<webhook>"}}\''                                                                                      |


<a id="cli-cmd-run"></a>
### `run` 

Use the `run` CLI command to execute a task and inject parameters by using a local or remote function.

Usage: mlrun [OPTIONS] URL [ARGS]...

Examples:
- `mlrun run -f db://getting-started-admin/prep-data --project getting-started-admin`
- `mlrun run -f myfunc.yaml -w -p p1=3`


| Flag                                           | Description                                                                                             |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------| 
| -p, &minus;&minus;param                        | Parameter name and value tuples; for example, `-p x=37 -p y='text'`                                     |
| -i, &minus;&minus;inputs                       | Input artifact; for example, `-i infile.txt=s3://mybucket/infile.txt`                                   |
| -o, &minus;&minus;outputs                      | Output artifact/result for kfp"                                                                         |
| &minus;&minus;in&minuspath                     | Base directory path/URL for storing input artifacts                                                     |
| &minus;&minus;out&minuspath                    | Base directory path/URL for storing output artifacts                                                    |
| -s, &minus;&minus;secrets                      | Secrets, either as `file=<filename>` or `env=<ENVAR>,...`; for example, `-s file=secrets.txt`           |
| &minus;&minus;uid                              | Unique run ID                                                                                           |
| &minus;&minus;name                             | Run name                                                                                                |
| &minus;&minus;workflow                         | Workflow name/id                                                                                        |
| &minus;&minus;project                          | Project name or ID                                                                                      |
| &minus;&minus;db                               | Save run results to DB url                                                                              |
| &minus;&minus;runtime                          | Function spec dict, for pipeline usage                                                                  |
| &minus;&minus;kfp                              | Running inside Kubeflow Piplines, do not use                                                            |
| -h, &minus;&minus;hyperparam                   | Hyper parameters (will expand to multiple tasks) e.g. --hyperparam p2=[1,2,3]                           |
| &minus;&minus;param&minus;file                 | Path to csv table of execution (hyper) params                                                           |
| &minus;&minus;selector                         | How to select the best result from a list, e.g. max.accuracy                                            |
| &minus;&minus;hyper&minus;param&minus;strategy | Hyperparam tuning strategy list, grid, random                                                           |
| &minus;&minus;hyper&minus;param&minus;options  | Hyperparam options json string                                                                          |
| -f, &minus;&minus;func&minus;url               | Path/URL of a YAML function-configuration file, or db://<project>/<name>[:tag] for a DB function object |
| &minus;&minus;task                             | Path/URL of a YAML task-configuration file                                                              |
| &minus;&minus;handler                          | Invoke the function handler inside the code file                                                        |
| &minus;&minus;mode                             | Special run mode ('pass' for using the command as is)                                                   |
| &minus;&minus;schedule                         | Cron schedule                                                                                           |
| &minus;&minus;from&minus;env                   | Read the spec from the env var                                                                          |
| &minus;&minus;dump                             | Dump run results as YAML                                                                                |
| &minus;&minus;image                            | Container image                                                                                         |
| &minus;&minus;kind                             | Serverless runtime kind                                                                                 |
| &minus;&minus;source                           | Source code archive/git                                                                                 |
| &minus;&minus;local                            | Run the task locally (ignore runtime)                                                                   |
| &minus;&minus;auto&minus;mount                 | Add volume mount to job using auto mount option                                                         |
| &minus;&minus;workdir                          | Run working directory                                                                                   |
| &minus;&minus;origin&minus;file                | For internal use                                                                                        |
| &minus;&minus;label                            | Run labels (key=val)                                                                                    |
| -w, &minus;&minus;watch                        | Watch/tail run log                                                                                      |
| &minus;&minus;verbose                          | Verbose log                                                                                             |
| &minus;&minus;scrape&minus;metrics             | Whether to add the `mlrun/scrape-metrics` label to this run's resources                                 |
| &minus;&minus;env&minus;file                   | Path to .env file to load config/variables from                                                         |
| &minus;&minus;auto&minus;build                 | When set functions will be built prior to run if needed                                                 |
| &minus;&minus;ensure&minus;project             | Ensure the project exists, if not, create project                                                       |
| &minus;&minus;returns                          | Logging configurations for the handler's returning values                                               |

<a id="cli-cmd-version"></a>
### `version` 
    
Use the `version` CLI command to get the mlrun server version.

<a id="cli-cmd-watch-stream"></a>
### `watch-stream` 

Use the `watch-stream` CLI command to watch a v3io stream and print data at a recurring interval.

Usage: mlrun watch-stream [OPTIONS] URL

Examples: 
- `mlrun watch-stream v3io:///users/my-test-stream`
- `mlrun watch-stream v3io:///users/my-test-stream -s 1`
- `mlrun watch-stream v3io:///users/my-test-stream -s 1 -s 2`
- `mlrun watch-stream v3io:///users/my-test-stream -s 1 -s 2 --seek EARLIEST`

| Flag                        | Description                                                |
|-----------------------------|------------------------------------------------------------| 
| -s, &minus;&minus;shard-ids | Shard id to listen on (can be multiple).                   |  
| &minus;&minus;seek          | Where to start/seek (EARLIEST or LATEST)                   |
| -i, &minus;&minus;interval  | Interval in seconds. Default = 3                           |
| -j, &minus;&minus;is-json   | Indicates that the payload is json (will be deserialized). |
    
<a id="git-func"></a>
## Building and running a function from a Git repository
 
To build and run a function from a Git repository, start out by adding a YAML function-configuration file in your local environment.
This file should describe the function and define its specification.
For example, create a **myfunc.yaml** file with the following content in your working directory:
```yaml
kind: job
metadata:
  name: remote-demo1
  project: ''
spec:
  command: 'examples/training.py'
  args: []
  image: .mlrun/func-default-remote-demo-ps-latest
  image_pull_policy: Always
  build:
    base_image: mlrun/mlrun:1.4.0
    source: git://github.com/mlrun/mlrun
```

Then, run the following CLI command and pass the path to your local function-configuration file as an argument to build the function's container image according to the configured requirements.
For example, the following command builds the function using the **myfunc.yaml** file from the current directory:
```sh
mlrun build myfunc.yaml
```

When the build completes, you can use the `run` CLI command to run the function.
Set the `-f` option to the path to the local function-configuration file, and pass the relevant parameters.
For example:
```sh
mlrun run -f myfunc.yaml -w -p p1=3
```

You can also try the following function-configuration example, which is based on the MLRun CI demo:
```yaml
kind: job
metadata:
  name: remote-git-test
  project: default
  tag: latest
spec:
  command: 'myfunc.py'
  args: []
  image_pull_policy: Always
  build:
    commands: []
    base_image: mlrun/mlrun:1.4.0
    source: git://github.com/mlrun/ci-demo.git
```

<a id="sources-archive"></a>
## Using a sources archive

The `-a|--archive` option of the CLI [`build`](#cli-cmd-build) command enables you to define a remote object path for storing TAR archive files with all the required code dependencies.
The remote location can be, for example, in an AWS S3 bucket or in a data container in an Iguazio MLOps Platform ("platform") cluster.
Alternatively, you can also set the archive path by using the `MLRUN_DEFAULT_ARCHIVE` environment variable.
When an archive path is provided, the remote builder archives the configured function sources (see the `-s|-source` [`build`](#cli-cmd-build) option) into a TAR archive file, and then extracts (untars) all of the archive files (i.e., the function sources) into the configured archive location.
<!-- [IntInfo] MLRUN_DEFAULT_ARCHIVE is referenced in the code using
  `mlconf.default_archive` when using `from .config import config as mlconf`.
-->

To use the archive option, first create a local function-configuration file.
For example, you can create a **function.yaml** file in your working directory with the following content; the specification describes the environment to use, defines a Python base image, adds several packages, and defines **examples/training.py** as the application to execute on `run` commands:
```yaml
kind: job
metadata:
  name: remote-demo4
  project: ''
spec:
  command: 'examples/training.py'
  args: []
  image_pull_policy: Always
  build:
    commands: []
    base_image: mlrun/mlrun:1.4.0
```

Next, run the following MLRun CLI command to build the function; replace the `<...>` placeholders to match your configuration:

```sh
mlrun build <function-configuration file path> -a <archive path/URL> [-s <function-sources path/URL>]
```

For example, the following command uses the **function.yaml** configuration file (`.`), relies on the default function-sources path (`./`), and sets the target archive path to `v3io:///users/$V3IO_USERNAME/tars`.
So, for a user named "admin", for example, the function sources from the local working directory will be archived and then extracted into an **admin/tars** directory in the "users" data container of the configured platform cluster (which is accessed via the `v3io` data mount):

```sh
mlrun build . -a v3io:///users/$V3IO_USERNAME/tars
```

> **Note:**
> - `.` is a shorthand for a **function.yaml** configuration file in the local working directory.
> - The `-a|--archive` option is used to instruct MLRun to create an archive file from the function-code sources at the location specified by the `-s|--sources` option; the default sources location is the current directory (`./`).

After the function build completes, you can run the function with some parameters.
For example:

```sh
mlrun run -f . -w -p p1=3
```
