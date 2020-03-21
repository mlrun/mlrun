# Using MLRun from a Remote Client

This tutorial explains how to use MLRun from a local development environment (IDE) to run jobs on a remote cluster.

#### In This Document

- [Prerequisites](#prerequisites)
- [CLI Commands](#cli-commands)
  - [The `build` Command](#cli-cmd-build)
  - [The `run` Command](#cli-cmd-run)
- [Building and Running a Function from a Git Repository](#git-func)
- [Using a Sources Archive](#sources-archive)

<a id="prerequisites"></a>
## Prerequisites

Before you begin, ensure that the following prerequisites are met:

1. Install MLRun locally.
    You can do this by running the following from a command line:
    ```sh
    pip install mlrun
    ```
2. Ensure that you have remote access to your MLRun service (i.e., to the service's NodePort on the remote Kubernetes cluster).
3. Set environment variables to define your MLRun configuration.
    As a minimum requirement &mdash;

    - Set `MLRUN_DBPATH` to the URL of the remote MLRun database/API service; replace the `<...>` placeholders to identify your remote target:

      ```sh
      MLRUN_DBPATH=http://<cluster IP>:<port>
      ```
    - If the remote service is on an instance of the Iguazio Data Science Platform (**"the platform"**), set the following environment variables as well; replace the `<...>` placeholders with the information for your specific platform cluster:

      ```sh
      V3IO_USERNAME=<username of a platform user with MLRun admin privileges>
      V3IO_API=<API endpoint of the web-APIs service endpoint; e.g., "webapi.default-tenant.app.mycluster.iguazio.com">
      V3IO_ACCESS_KEY=<platform access key>
      ```

<a id="cli-commands"></a>
## CLI Commands

Use the following commands of the MLRun command-line interface (CLI) &mdash; `mlrun` &mdash; to build and run MLRun functions:

- [`build`](#cli-cmd-build)
- [`run`](#cli-cmd-run)

<a id="cli-cmd-build"></a>
### The `build` Command

Use the `build` CLI command to build all the function dependencies from the function specification into a function container (Docker image).
This command supports many options, including the following; for the full list, run `mlrun build --help`:

```sh
  --name TEXT            Function name
  --project TEXT         Project name
  --tag TEXT             Function tag
  -i, --image TEXT       Target image path
  -s, --source TEXT      Path/URL of the function source code - a PY file, or a directory to archive
                         when using the -a|--archive option (default: './')
  -b, --base-image TEXT  Base Docker image
  -c, --command TEXT     Build commands; for example, '-c pip install pandas'
  --secret-name TEXT     Name of a container-registry secret
  -a, --archive TEXT     Path/URL of a target function-sources archive directory: as part of
                         the build, the function sources (see -s|--source) are archived into a
                         TAR file and then extracted into the archive directory 
  --silent               Don't show build logs
  --with-mlrun           Add the MLRun package ("mlrun")
```

> **Note:** For information about using the `-a|--archive` option to create a function-sources archive, see [Using a Sources Archive](#sources-archive) later in this tutorial.

<a id="cli-cmd-run"></a>
### The `run` Command

Use the `run` CLI command to execute a task by using a local or remote function.
This command supports many options, including the following; for the full list, run `mlrun run --help`:

```sh
  -p, --param key=val    Parameter name and value tuples; for example, `-p x=37 -p y='text'`
  -i, --inputs key=path  Input artifact; for example, `-i infile.txt=s3://mybucket/infile.txt`
  --in-path TEXT         Base directory path/URL for storing input artifacts
  --out-path TEXT        Base directory path/URL for storing output artifacts
  -s, --secrets TEXT     Secrets, either as `file=<filename>` or `env=<ENVAR>,...`; for example, `-s file=secrets.txt`
  --name TEXT            Run name
  --project TEXT         Project name or ID
  -f, --func-url TEXT    Path/URL of a YAML function-configuration file, or db://<project>/<name>[:tag] for a DB function object
  --task TEXT            Path/URL of a YAML task-configuration file
  --handler TEXT         Invoke the function handler inside the code file
```

<a id="git-func"></a>
## Building and Running a Function from a Git Repository

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
    #commands: ['pip install pandas']
    base_image: mlrun/mlrun:dev
    source: git://github.com/mlrun/mlrun
```

Then, run the following CLI command and pass the path to your local function-configuration file as an argument to build the function's container image according to the configured requirements.
For example, the following command builds the function using the **myfunc.yaml** file from the current directory:
```sh
mlrun build myfunc.yaml
```

When the build completes, you can use the `run` CLI command to run the function.
Set the `-f` option to the path to the local function-configuration file, and pass relevant parameters.
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
    commands: ['pip install pandas']
    base_image: mlrun/mlrun:dev
    source: git://github.com/mlrun/ci-demo.git
```

<a id="sources-archive"></a>
## Using a Sources Archive

The `-a|--archive` option of the CLI [`build`](#cli-cmd-build) command enables you to define a remote object path for storing TAR archive files with all the required code dependencies.
The remote location can be, for example, in an AWS S3 bucket or in a data container in an Iguazio Data Science Platform ("platform") cluster.
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
    commands: ['pip install mlrun pandas']
    base_image: python:3.6-jessie
```

Next, run the following MLRun CLI command to build the function; replace the `<...>` placeholders to match your configuration:
```sh
mlrun build <function-configuration file path> -a <archive path/URL> [-s <function-sources path/URL>]
```
> **Note:**
> - `.` is a shorthand for a **function.yaml** configuration file in the local working directory.
> - The `-a|--archive` option is used to instruct MLRun to create an archive file from the function-code sources at the location specified by the `-s|--sources` option; the default sources location is the current directory (`./`).

For example, the following command uses the **function.yaml** configuration file (`.`), relies on the default function-sources path (`./`), and sets the target archive path to `v3io:///users/$V3IO_USERNAME/tars`.
So, for a user named "admin", for example, the function sources from the local working directory will be archived and then extracted into an **admin/tars** directory in the "users" data container of the configured platform cluster (which is accessed via the `v3io` data mount):
```sh
mlrun build . -a v3io:///users/$V3IO_USERNAME/tars
```

After the function build completes, you can run the function with some parameters.
For example:
```sh
mlrun run -f . -w -p p1=3
```

