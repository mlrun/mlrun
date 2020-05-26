# Quick-Start <a id="top"/></a> <!-- omit in toc -->

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
``` bash
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
```bash
pip install mlrun
```

Set-up an artifacts path and the MLRun Database path


```python
from os import path
from mlrun import mlconf

# Target location for storing pipeline artifacts
artifact_path = path.abspath('jobs')
# MLRun DB path or API service URL
mlconf.dbpath = mlconf.dbpath or 'http://mlrun-api:8080'

print(f'Artifacts path: {artifact_path}\nMLRun DB path: {mlconf.dbpath}')
```

    Artifacts path: /User/mlrun/jobs
    MLRun DB path: http://10.193.140.11:8080


<a id="projects"></a>
## Projects

Projects in the platform are used to package multiple functions, workflows, and artifacts. Projects are created by using the `new_project` MLRun method.
Projects are visible in the MLRun dashboard only after they're saved to the MLRun database, which happens whenever you run code for a project.

For example, use the following code to create a project named **my-project** and stores the project definition in a subfolder named `conf`:


```python
from mlrun import new_project

project_name = 'my-project'
project_path = path.abspath('conf')
project = new_project(project_name, project_path, init_git=True)

print(f'Project path: {project_path}\nProject name: {project_name}')
```

    Project path: /User/mlrun/conf
    Project name: my-project


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

For this purpose, we'll add a `context` parameter which will be used to log our artifacts. Our code will now look as follows:


```python
def get_data(context, source_url, format='csv'):

    df = source_url.as_df()

    target_path = path.join(context.artifact_path, 'data')
    # Store the data set in your artifacts database
    context.log_dataset('source_data', df=df, format=format,
                        index=False, artifact_path=target_path)
```

<a id="run-local-code"></a>
## Run Local Code

As input, we will provide a CSV file from S3:


```python
# Set the source-data URL
source_url = 'http://iguazio-sample-data.s3.amazonaws.com/iris_dataset.csv'
```

Next you can call this function locally, using the `run_local` method. This is a wrapper that will store the execution results in the MLRun database.

```python
from mlrun import run_local
get_data_run = run_local(name='get_data',
                         handler=get_data,
                         inputs={'source_url': source_url},
                         project=project_name, artifact_path=artifact_path)
```

This will produce a simlar output: 

    [mlrun] 2020-05-26 19:19:47,286 starting run get_data uid=ccadebc11f024aa88d63965fdc223c5f  -> http://10.193.140.11:8080
    [mlrun] 2020-05-26 19:19:47,963 log artifact source_data at /User/mlrun/jobs/data/source_data.csv, size: 2776, db: Y

    [mlrun] 2020-05-26 19:19:47,987 run executed, status=completed

If you run the function in a Jupyter notebook, the output cell for your function execution will contain a table with run information &mdash; including the state of the execution, all inputs and parameters, and the execution results and artifacts.
Click on the `source_data` artifact in the **artifacts** column to see a short summary of the data set, as illustrated in the following image:
<br><br>
![MLRun quick start get data output](_static/images/mlrun-quick-start-get-data-output.png)

<a id="experiment-tracking-ui"></a>
## Experiment Tracking UI

<br><br>
<img src="_static/images/mlrun-quick-start-get-data-output-ui-artifacts.png" alt="ui-artifacts" width="800"/>


<br><br>
<img src="_static/images/mlrun-quick-start-get-data-output-ui-statistics.png" alt="ui-statistics" width="800"/>

<a id="running-functions-on-different-runtimes"></a>
## Running functions on different runtimes

<a id="pipelines"></a>
## Pipelines

<a id="functions-marketplace"></a>
## Functions marketplace

Before implementing your own functions, you should first take a look at the [**MLRun functions marketplace** GitHub repository](https://github.com/mlrun/functions/). The marketplace is a centralized location for open-source contributions of function components that are commonly used in machine-learning development.

For example, the [`describe` function](https://github.com/mlrun/functions/blob/master/describe/describe.ipynb) visualizes the data by creating a histogram, imbalance and correlation matrix plots. 

Use the `set_function` MLRun project method, which adds or updates a function object in a project, to load the `describe` marketplace function into a new `describe` project function.


```python
project.set_function('hub://describe', 'describe')
```

You can then run the function as part of your project, just as any other function that you have written yourself.
