<a id="top"></a>
# Quick-Start <!-- omit in toc -->

- [Installation](#installation)
- [MLRun Setup](#mlrun-setup)
- [Projects](#projects)
- [Experiment Tracking](#experiment-tracking)
- [Run Local Code](#run-local-code)
- [Experiment Tracking UI](#experiment-tracking-ui)
- [Running functions on different runtimes](#running-functions-on-different-runtimes)
- [Pipelines](#pipelines)
- [Functions marketplace](#functions-marketplace)


<a id="installation"></a>
## Installation

MLRun requires separate containers for the API and the dashboard (UI).

To install and run MLRun locally using Docker
```
MLRUN_IP=localhost
SHARED_DIR=/home/me/data
# On Windows, use host.docker.internal for MLRUN_IP

docker pull quay.io/iguazio/mlrun-ui:latest
docker pull mlrun/jupy:latest

docker run -it -p 4000:80 --rm -d --name mlrun-ui -e MLRUN_API_PROXY_URL=http://${MLRUN_IP}:8080 quay.io/iguazio/mlrun-ui:latest
docker run -it -p 8080:8080 -p 8888:8888 --rm -d --name jupy -v $(SHARED_DIR}:/home/jovyan/data mlrun/jupy:latest
```

Using Docker only supports local runs. To fully leverage MLRun capabilities with different runtimes on top of Kubernetes, refer to the Kuberenetes installation instructions in the [Installation Guide](install.html#k8s-cluster)

<a id="setup"></a>
## MLRun Setup

Run the following command from your Python development environment (such as Jupyter Notebook) to install the MLRun package (`mlrun`), which includes a Python API library and the `mlrun` command-line interface (CLI):
```python
pip install mlrun
```

Set-up an artifacts path and the MLRun Database path

``` python
from mlrun import mlconf

# Target location for storing pipeline artifacts
artifact_path = path.abspath('jobs')
# MLRun DB path or API service URL
mlconf.dbpath = mlconf.dbpath or 'http://mlrun-api:8080'

print(f'Artifacts path: {artifact_path}\nMLRun DB path: {mlconf.dbpath}')
``` 

<a id="projects"></a>
## Projects

Projects in the platform are used to package multiple functions, workflows, and artifacts. Projects are created by using the `new_project` MLRun method.
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


<a id="experiment-tracking"></a>
## Experiment Tracking

MLRun introduces the concept of functions, and these functions are part of the project. If you have existing code, the first thing to do is to integrate this code with MLRun. This will not just allow you to run your code in different runtimes, but also enable you to track the function calls, with their inputs and results.

Let's take a simple scenario. First you have some code that reads either a csv file or parquet and returns a DataFrame.

```python
import pandas as pd

# Ingest a data set into the platform
def get_data(source_url):

    if source_url.endswith(".csv"):
        df = pd.read_csv(source_url)
    elif source_url.endswith(".parquet") or source_url.endswith(".pq"):
        df = pd.read_parquet(source_url)
    else:
        raise Exception(f"file type unhandled {source_url}")

    return df
```

We would like to do 2 things:
1. Have MLRun handle the data read
2. Log this data to the MLRun database

For this purpose, we'll add a context parameter which will be used to log our artifacts. Our code will now look as follows:

``` python
def get_data(context, source_url, format='csv'):

    df = source_url.as_df()

    target_path = path.join(context.artifact_path, 'data')
    # Store the data set in your artifacts database
    context.log_dataset('source_data', df=df, format=format,
                        index=False, artifact_path=target_path)
```

<a id="run-local-code"></a>
## Run Local Code

Next you can call this function locally, using the `run_local` method

``` python
from mlrun import run_local
get_data_run = run_local(name='get_data',
                         handler=get_data,
                         inputs={'source_url': source_url},
                         project=project_name, artifact_path=artifact_path)
```

<a id="experiment-tracking-ui"></a>
## Experiment Tracking UI

<a id="running-functions-on-different-runtimes"></a>
## Running functions on different runtimes

<a id="pipelines"></a>
## Pipelines

<a id="functions-marketplace"></a>
## Functions marketplace
