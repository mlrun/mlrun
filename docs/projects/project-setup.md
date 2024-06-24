(project-setup)=
# MLRun project bootstrapping with `project_setup.py`

## Overview

The `project_setup.py` script in MLRun automates project initialization and configuration, facilitating seamless setup of MLRun projects by registering functions, workflows, Git sources, Docker images, and more. It ensures consistency by registering and updating all functions and workflows within the project.

Upon loading an MLRun project via {py:meth}`~mlrun.projects.get_or_create_project` or {py:meth}`~mlrun.projects.load_project`, the system automatically invokes the `project_setup.py` script.

**Note:** Ensure the script resides in the root of the project context.

```python
import mlrun

# Load or create an MLRun project
project = mlrun.get_or_create_project(
    "my-project"
)  # project_setup.py called while loading project
```

## Format
The `project_setup.py` script returns the updated MLRun project after applying the specified configurations. It should have a `setup` function which receives an {py:class}`~mlrun.projects.MlrunProject` and returns an {py:class}`~mlrun.projects.MlrunProject`.

```python
def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    # ... (setup configurations)

    # Save and return the project:
    project.save()
    return project
```

## Example Usage

Here's an example directory structure of a project utilizing the `project_setup.py` script:
```
.
├── .env
└── src
    ├── functions
    │   ├── data.py
    │   └── train.py
    ├── project_setup.py
    └── workflows
        └── main_workflow.py
```

The `project_setup.py` script looks like the following:
```python
import os

import mlrun


def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    source = project.get_param("source")
    secrets_file = project.get_param("secrets_file")
    default_image = project.get_param("default_image")

    # Set project git/archive source and enable pulling latest code at runtime
    if source:
        print(f"Project Source: {source}")
        project.set_source(project.get_param("source"), pull_at_runtime=True)

    # Create project secrets and also load secrets in local environment
    if secrets_file and os.path.exists(secrets_file):
        project.set_secrets(file_path=secrets_file)
        mlrun.set_env_from_file(secrets_file)

    # Set default project docker image - functions that do not specify image will use this
    if default_image:
        project.set_default_image(default_image)

    # MLRun Functions - note that paths are relative to the project context (./src)
    project.set_function(
        name="get-data",
        func="functions/data.py",
        kind="job",
        handler="get_data",
    )

    project.set_function(
        name="train",
        func="functions/train.py",
        kind="job",
        handler="train_model",
    )

    # MLRun Workflows - note that paths are relative to the project context (./src)
    project.set_workflow("main", "workflows/main_workflow.py")

    # Save and return the project:
    project.save()
    return project
```

The project can then be loaded using the following code snippet:
```python
project = mlrun.get_or_create_project(
    name="my-project",
    context="./src",  # project_setup.py should be in this directory
    parameters={
        "source": "https://github.com/mlrun/my-repo#main",
        "secrets_file": ".env",
        "default_image": "mlrun/mlrun",
    },
)
```

## Common Operations

Some common operations that can be added to the `project_setup.py` script include:


### Set Project Source

Set the project source and enable pulling at runtime if specified. See {py:meth}`~mlrun.projects.MlrunProject.set_source` for more info.

```python
source = project.get_param("source")  # https://github.com/mlrun/my-repo#main

project.set_source(source, pull_at_runtime=True)
```

### Export Project to Zip File Archive
Export the local project directory contents to a zip file archive. Use this in conjunction with setting the project source for rapid iteration without requiring a Git commit for each change. See {py:meth}`~mlrun.projects.MlrunProject.set_source` and {py:meth}`~mlrun.projects.MlrunProject.export` for more info.

**Note:** This requires using the Iguazio `v3io` data layer or some `s3` compliant object storage such as `minio`.

```python
source = project.get_param("source")  # v3io:///bigdata/my_project.zip

project.set_source(source, pull_at_runtime=True)
if ".zip" in source:
    print(f"Exporting project as zip archive to {source}...")
    project.export(source)
```

### Set Existing Default Project Image
Define the default Docker image for the project. It will be used for functions without a specified image. See {py:meth}`~mlrun.projects.MlrunProject.set_default_image` for more info.

```python
default_image = project.get_param("default_image")  # mlrun/mlrun

if default_image:
    project.set_default_image(default_image)
```

### Build a Docker Image
Build a Docker image and optionally set it as the project default. See {py:meth}`~mlrun.projects.MlrunProject.build_image` for more info.

```python
base_image = project.get_param("base_image")  # mlrun/mlrun
requirements_file = project.get_param("requirements_file")  # requirements.txt

project.build_image(
    base_image=base_image, requirements_file=requirements_file, set_as_default=True
)
```

### Register Functions
Register MLRun functions within the project, specifying their names, associated files, kind (e.g., job), and handlers. See {py:meth}`~mlrun.projects.MlrunProject.set_function` for more info.

```python
project.set_function(
    name="get-data",
    func="data.py",
    kind="job",
    handler="get_data",
)
```

### Define Workflows
Define MLRun workflows within the project, associating them with specific files. See {py:meth}`~mlrun.projects.MlrunProject.set_workflow` for more info.

```python
project.set_workflow("main", "main_workflow.py")
```

### Manage Secrets
Create project secrets by setting them from a specified file path and load them as environment variables in the local environment. See {py:meth}`~mlrun.projects.MlrunProject.set_secrets` and {py:meth}`~mlrun.set_env_from_file` for more info.

```python
secrets_file = project.get_param("secrets_file")  # .env

if secrets_file and os.path.exists(secrets_file):
    project.set_secrets(file_path=secrets_file)
    mlrun.set_env_from_file(secrets_file)
```

### Register Project Artifacts
Register artifacts like models or datasets in the project. Useful for version control and transferring artifacts between environments (e.g. dev, staging, prod) via CI/CD. See {py:meth}`~mlrun.projects.MlrunProject.set_artifact` and {py:meth}`~mlrun.projects.MlrunProject.register_artifacts` for more info.

```python
project.set_artifact(
    key="model",
    artifact="artifacts/model:challenger.yaml",  # YAML file in project directory
    tag="challenger",
)
project.register_artifacts()
```

### Defining K8s Resource Requirements for Functions
Add Kubernetes resources by setting requests/limits for a given MLRun function. See [CPU, GPU, and memory limits for user jobs](../runtimes/configuring-job-resources.html#cpu-gpu-and-memory-limits-for-user-jobs) for more info.

```python
gpus = project.get_param("num_gpus_per_replica") or 4
cpu = project.get_param("num_cpus_per_replica") or 48
mem = project.get_param("memory_per_replica") or "192Gi"

train_function = project.set_function(
    "trainer.py",
    name="training",
    kind="job",
)
train_function.with_limits(gpus=gpus, cpu=cpu, mem=mem)
train_function.save()
```

### Loading a project from a template
You can load a project from a template only if you make one of these changes:
1. Set the allow_cross_project flag = True and change the name of the project.
2. Change the name in the yaml file or delete the file.
3. Change the context dir.

```python
import mlrun

project = mlrun.load_project(
    name="my-project",
    context="./src",  # assuming here there is a project.yaml with name that is not my-project
    allow_cross_project=True,
)
```

**Note:** This is relevant also for the `get_or_create_project` function.