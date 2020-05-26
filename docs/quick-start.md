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

<a id="experiment-tracking"></a>
## Experiment Tracking

<a id="run-local-code"></a>
## Run Local Code

<a id="experiment-tracking-ui"></a>
## Experiment Tracking UI

<a id="running-functions-on-different-runtimes"></a>
## Running functions on different runtimes

<a id="pipelines"></a>
## Pipelines

<a id="functions-marketplace"></a>
## Functions marketplace
