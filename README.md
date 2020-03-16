<a id="top"></a>
# MLRun

[![CircleCI](https://circleci.com/gh/mlrun/mlrun/tree/development.svg?style=svg)](https://circleci.com/gh/mlrun/mlrun/tree/development)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version fury.io](https://badge.fury.io/py/mlrun.svg)](https://pypi.python.org/pypi/mlrun/)
[![Documentation](https://readthedocs.org/projects/mlrun/badge/?version=latest)](https://mlrun.readthedocs.io/en/latest/?badge=latest)

MLRun is a generic and convenient mechanism for data scientists and software developers to describe and run tasks related to machine learning (ML) in various, scalable runtime environments and ML pipelines while automatically tracking executed code, metadata, inputs, and outputs.
MLRun integrates with the [Nuclio](https://nuclio.io/) serverless project and with [Kubeflow Pipelines](https://github.com/kubeflow/pipelines).

MLRun features a Python package (`mlrun`), a command-line interface (`mlrun`), and a graphical user interface (the MLRun dashboard).

#### In This Document
- [General Concept and Motivation](#concepts-n-motivation)
- [Installation](#installation)
- [Examples and Tutorial Notebooks](#examples-n-tutorial-notebooks)
- [Quick-Start Tutorial &mdash; Architecture and Usage Guidelines](#qs-tutorial)

<a id="concepts-n-motivation"></a>
## General Concept and Motivation
- [The Challenge](#the-challenge)
- [The MLRun Vision](#the-vision)

<a id="the-challenge"></a>
### The Challenge

As an ML developer or data scientist, you typically want to write code in your preferred local development environment (IDE) or web notebook, and then run the same code on a larger cluster using scale-out containers or functions.
When you determine that the code is ready, you or someone else need to transfer the code to an automated ML workflow (for example, using [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-quickstart/)).
This pipeline should be secure and include capabilities such as logging and monitoring, as well as allow adjustments to relevant components and easy redeployment.

However, the implementation is challenging: various environments (**"runtimes"**) use different configurations, parameters, and data sources.
In addition, multiple frameworks and platforms are used to focus on different stages of the development life cycle.
This leads to constant development and DevOps/MLOps work.

Furthermore, as your project scales, you need greater computation power or GPUs, and you need to access large-scale data sets.
This cannot work on laptops.
You need a way to seamlessly run your code on a remote cluster and automatically scale it out.

<a id="the-vision"></a>
### The MLRun Vision

When ML running experiments, you should ideally be able to record and version your code, configuration, outputs, and associated inputs (lineage), so you can easily reproduce and explain your results.
The fact that you probably need to use different types of storage (such as files and AWS S3 buckets) and various databases, further complicates the implementation.

Wouldn't it be great if you could write the code once, using your preferred development environment and simple "local" semantics, and then run it as-is on different platforms?
Imagine a layer that automates the build process, execution, data movement, scaling, versioning, parameterization, outputs tracking, and more.
A world of easily developed, published, or consumed data or ML "functions" that can be used to form complex and large-scale ML pipelines.

In addition, imagine a marketplace of ML functions that includes both open-source templates and your internally developed functions, to support code reuse across projects and companies and thus further accelerate your work.

<b>This is the goal of MLRun.</b>

> **Note:** The code is in early development stages and is provided as a reference.
> The hope is to foster wide industry collaboration and make all the resources pluggable, so that developers can code to a single API and use various open-source projects or commercial products.

[Back to top](#top)

<a id="installation"></a>
## Installation

Run the following command from your Python development environment (such as Jupyter Notebook) to install the MLRun package (`mlrun`), which includes a Python API library and the `mlrun` command-line interface (CLI):
```python
pip install mlrun
```

MLRun requires separate containers for the API and the dashboard (UI).
You can also select to use the pre-baked JupyterLab image.

To install and run MLRun locally using Docker or Kubernetes, see the instructions in [**hack/local/README.md**](hack/local/README.md).

<a id="installation-iguazio-platform"></a>
### Installation on the Iguazio Data Science Platform

To install MLRun on an instance of the [Iguazio Data Science Platform](https://www.iguazio.com) (**"the platform"**) &mdash;

1. Create a copy of the [**hack/mlrun-all.yaml**](hack/mlrun-all.yaml) configuration file; you can also rename your copy.
    You can fetch the file from GitHub by running the following from a command line; to install a specific version of MLRun, replace `master` with the relevant version tag:
    ```sh
    curl -O https://raw.githubusercontent.com/mlrun/mlrun/master/hack/mlrun-all.yaml
    ```
    <!-- [c-mlrun-versions] TODO: When there are MLRun version tags, instruct
      to replace `master` with the version tag for the MLRun version supported
      for the current platform version. -->

2. Edit the configuration file to match your environment and desired configuration.
    The following is required:

    - Replace all `<...>` placeholders in the file.
        Be sure to replace `<access key>` with a valid platform access key and `<default Docker registry URL>` with the URL of the default Docker Registry service of your platform cluster.

        > **Note:** In platform cloud deployments, the URL of the default Docker Registry service is `docker-registry.default-tenant.<cluster DNS>:80`.
        > Note the port number (80), which indicates a local on-cluster registry (unlike the default Docker port number).
    - Uncomment the `volumes` and the`mlrun-api` container's `volumeMounts` configurations to add a volume for persisting data in the platform's data store (using the `v3io` data mount).
    - Ensure that the value of the `V3IO_USERNAME` environment variable (`env`) and the `volumes.subPath` field are set to the name of a platform user with MLRun admin privileges (default: "admin").

3. When you're ready, install MLRun by running the following from a platform command-line shell; replace `<namespace>` with your cluster's Kubernetes namespace, and `<configuration file>` with the path to your edited configuration file:
    ```sh
    kubectl apply -n <namespace> -f <configuration file>
    ```

[Back to top](#top)

<a id="examples-n-tutorial-notebooks"></a>
## Examples and Tutorial Notebooks

MLRun has many code examples and tutorial Jupyter notebooks with embedded documentation, ranging from examples of basic tasks to full end-to-end use-case applications, including the following; note that some of the examples are found in other mlrun GitHub repositories:

- Learn MLRun basics &mdash; [**examples/mlrun_basics.ipynb**](examples/mlrun_basics.ipynb)
- Convert local runs to Kubernetes jobs and create automated pipelines in a single notebook &mdash; [**examples/mlrun_jobs.ipynb**](examples/mlrun_jobs.ipynb)
- End-to-end XGBoost pipeline, including data ingestion, model training, verification, and deployment &mdash; [**demo-xgb-project**](https://github.com/mlrun/demo-xgb-project) repo
- MLRun with scale-out runtimes &mdash;
  - Distributed TensorFlow with Horovod and MPIJob &mdash; [**examples/mlrun_mpijob_classify.ipynb**](examples/mlrun_mpijob_classify.ipynb)
  - Serverless model serving with Nuclio &mdash; [**examples/xgb_serving.ipynb**](examples/xgb_serving.ipynb)
  - Dask &mdash; [**examples/mlrun_dask.ipynb**](examples/mlrun_dask.ipynb)
  - Spark &mdash; [**examples/mlrun_sparkk8s.ipynb**](examples/mlrun_sparkk8s.ipynb)
- MLRun projects &mdash;
  - Load a project from a remote Git location and run pipelines &mdash; [**examples/load-project.ipynb**](examples/load-project.ipynb)
  - Create a new project, functions, and pipelines, and upload to Git &mdash; [**examples/new-project.ipynb**](examples/new-project.ipynb)
- Import and export functions using files or Git &mdash; [**examples/mlrun_export_import.ipynb**](examples/mlrun_export_import.ipynb)
- Query the MLRun DB &mdash; [**examples/mlrun_db.ipynb**](examples/mlrun_db.ipynb)

<a id="additional-examples"></a>
### Additional Examples

- Deep-learning pipeline (full end-to-end application), including data collection and labeling, model training and serving, and implementation of an automated workflow &mdash; [mlrun/demo-image-classification](https://github.com/mlrun/demo-image-classification) repo
- Additional end-to-end use-case applications &mdash; [mlrun/demos](https://github.com/mlrun/demos) repo
- MLRun functions Library &mdash; [mlrun/functions](https://github.com/mlrun/functions) repo [WORK IN PROGRESS]

[Back to top](#top)

<a id="qs-tutorial"></a>
## Quick-Start Tutorial &mdash; Architecture and Usage Guidelines
<!-- TODO: Move this to a separate docs/quick-start.md file, add an opening
  paragraph, update the heading levels, add a `top` anchor, and remove the
  "Back to quick-start TOC" links (leaving only the "Back to top" links). -->

- [Basic Components](#basic-components)
- [Managed and Portable Execution ](#managed-and-portable-execution)
- [Automated Code Deployment and Containerization](#auto-parameterization-artifact-tracking-n-logging)
- [Using Hyperparameters for Job Scaling](#using-hyperparameters-for-job-scaling)
- [Automated Code Deployment and Containerization](#auto-code-deployment-n-containerization)
- [Build and run function from a remote IDE using the CLI](examples/remote.md)
- [Running an ML Workflow with Kubeflow Pipelines](#run-ml-workflow-w-kubeflow-pipelines)
- [Viewing Run Data and Performing Database Operations](#db-operations)
  - [The MLRun Dashboard](#mlrun-ui)
  - [MLRun Database Methods](#mlrun-db-methods)
- [Additional Information and Examples](#additional-info-n-examples)
  - [Replacing Runtime Context Parameters from the CLI](#replace-runtime-context-param-from-cli)
  - [Remote Execution](#remote-execution)
- [Running an MLRun Service](#run-mlrun-service)
  - [Using Docker](#run-mlrun-service-docker)
  - [Using the MLRun CLI](#run-mlrun-service-cli)

<a id="basic-components"></a>
### Basic Components

MLRun has the following main components, which are usually grouped into **"projects"**:

- <a id="def-function"></a>**Function** &mdash; a software package with one or more methods and runtime-specific attributes (such as image, command, arguments, and environment).
    A function can run one or more runs or tasks, it can be created from templates, and it can be stored in a versioned database.
- <a id="def-task"></a>**Task** &mdash; defines the parameters, inputs, and outputs of a logical job or task to execute.
    A task can be created from a template, and can run over different runtimes or functions.
- <a id="def-run"></a>**Run** &mdash; contains information about an executed task.
  The run object is created as a result of running a task on a function, and it has all the attributes of a task (such as run parameters and relevant inputs and outputs) with the addition of the execution status and results (including links to output artifacts).
- <a id="def-artifact"></a>**Artifact** &mdash; versioned data artifacts (such as files, objects, data sets, and models) that are produced or consumed by functions, runs, and workflows.
- <a id="def-workflow"></a>**Workflow** &mdash; defines a functions pipeline or a directed acyclic graph (DAG) to execute using Kubeflow Pipelines.

<a id="managed-and-portable-execution"></a>
### Managed and Portable Execution

<a id="def-runtime"></a>MLRun supports various types of **"runtimes"** &mdash; computation frameworks such as local, Kubernetes job, Dask, Nuclio, Spark, or MPI job (Horovod).
Runtimes may support parallelism and clustering to distribute the work among multiple workers (processes/containers).

The following code example creates a task that defines a run specification &mdash; including the run parameters, inputs, and secrets.
You run the task on a "job" function, and print the result output (in this case, the "model" artifact) or watch the run's progress.
For more information and examples, see the [**examples/mlrun_basics.ipynb**](examples/mlrun_basics.ipynb) notebook.
```python
# Create a task and set its attributes
task = NewTask(handler=handler, name='demo', params={'p1': 5})
task.with_secrets('file', 'secrets.txt').set_label('type', 'demo')

run = new_function(command='myfile.py', kind='job').run(task)
run.logs(watch=True)
run.show()
print(run.artifact('model'))
```

You can run the same [task](#def-task) on different functions &mdash; enabling code portability, re-use, and AutoML.
You can also use the same [function](#def-function) to run different tasks or parameter combinations with minimal coding effort.

Moving from local notebook execution to remote execution &mdash; such as running a container job, a scaled-out framework, or an automated workflow engine like Kubeflow Pipelines &mdash; is seamless: just swap the runtime function or wire functions in a graph.
Continuous build integration and deployment (CI/CD) steps can also be configured as part of the workflow, using the `deploy_step` function method.

Functions (function objects) can be created by using any of the following methods:

- **`new_function`** &mdash; creates a function "from scratch" or from another function.
- **`code_to_function`** &mdash; creates a function from local or remote source code or from a web notebook.
- **`import_function`** &mdash; imports a function from a local or remote YAML function-configuration file or from a function object in the MLRun database (using a DB address of the format `db://<project>/<name>[:<tag>]`).

You can use the `save` function method to save a function object in the MLRun database, or the `export` method to save a YAML function-configuration function to your preferred local or remote location.
For function-method details and examples, see the embedded documentation/help text.

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="auto-parameterization-artifact-tracking-n-logging"></a>
### Automated Parameterization, Artifact Tracking, and Logging

After running a job, you need to be able to track it, including viewing the run parameters, inputs, and outputs.
To support this, MLRun introduces a concept of a runtime **"context"**: the code can be set up to get parameters and inputs from the context, as well as log run outputs, artifacts, tags, and time-series metrics in the context.

<a id="auto-parameterization-artifact-tracking-n-logging-example"></a>
#### Example

The following code example from the [**train-xgboost.ipynb**](https://github.com/mlrun/demo-xgb-project/blob/master/notebooks/train-xgboost.ipynb) notebook of the MLRun XGBoost demo (**demo-xgboost**) defines two functions:
the `iris_generator` function loads the Iris data set and saves it to the function's context object; the `xgb_train` function uses XGBoost to train an ML model on a data set and saves the log results in the function's context:

```python
import xgboost as xgb
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from mlrun.artifacts import TableArtifact, PlotArtifact
import pandas as pd


def iris_generator(context):
    iris = load_iris()
    iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_labels = pd.DataFrame(data=iris.target, columns=['label'])
    iris_dataset = pd.concat([iris_dataset, iris_labels], axis=1)
    context.logger.info('Saving Iris data set to "{}"'.format(context.out_path))
    context.log_artifact(TableArtifact('iris_dataset', df=iris_dataset))


def xgb_train(context,
              dataset='',
              model_name='model.bst',
              max_depth=6,
              num_class=10,
              eta=0.2,
              gamma=0.1,
              steps=20):

    df = pd.read_csv(dataset)
    X = df.drop(['label'], axis=1)
    y = df['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    # Get parameters from event
    param = {"max_depth": max_depth,
             "eta": eta, "nthread": 4,
             "num_class": num_class,
             "gamma": gamma,
             "objective": "multi:softprob"}

    xgb_model = xgb.train(param, dtrain, steps)

    preds = xgb_model.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    context.log_result('accuracy', float(accuracy_score(Y_test, best_preds)))
    context.log_artifact('model', body=bytes(xgb_model.save_raw()),
                         local_path=model_name, labels={'framework': 'xgboost'})
```

The example training function can be executed locally with parameters, and the run results and artifacts can be logged automatically into a database by using a single command, as demonstrated in the following example; the example sets the function's `eta` parameter:
```python
train_run = new_function().run(handler=xgb_train).with_params(eta=0.3)
```

Alternatively, you can replace the function with a serverless runtime to run the same code on a remote cluster, which could result in a ~10x performance boost.
You can find examples for different runtimes &mdash; such as a Kubernetes job, Nuclio, Dask, Spark, or an MPI job &mdash; in the MLRun [**examples**](examples) directory.

If you run your code from the `main` function, you can get the runtime context by calling the `get_or_create_ctx` method, as demonstrated in the following code from the MLRun [**training.py**](examples/training.py) example application.
The code also demonstrates how you can use the context object to read and write execution metadata, parameters, secrets, inputs, and outputs:

```python
from mlrun import get_or_create_ctx
from mlrun.artifacts import ChartArtifact, TableArtifact


def my_job(context, p1=1, p2='x'):
    # Load the MLRun runtime context. The context is set by the runtime
    # framework - for example, Kubeflow.

    # Access runtime-context information - input metadata, parameter values,
    # authentication secret (access key), and input artifacts (files)
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(context.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(context.get_input('infile.txt', 'infile.txt')
          .get()))

    # TODO: Run some useful code, such as ML training or data preparation.

    # Log scalar result values (job-result metrics)
    context.log_result('accuracy', p1 * 2)
    context.log_result('loss', p1 * 3)
    context.set_label('framework', 'sklearn')

    # Log various types of artifacts (file, web page, table), which will be
    # versioned and visible on the MLRun dashboard
    context.log_artifact('model', body=b'abc is 123', local_path='model.txt', labels={'framework': 'xgboost'})
    context.log_artifact('html_result', body=b'<b> Some HTML <b>', local_path='result.html')
    context.log_artifact(TableArtifact('dataset', '1,2,3\n4,5,6\n', visible=True,
                                        header=['A', 'B', 'C']), local_path='dataset.csv')

    # Create a chart output, which will be visible in the Kubeflow Pipelines UI
    chart = ChartArtifact('chart')
    chart.labels = {'type': 'roc'}
    chart.header = ['Epoch', 'Accuracy', 'Loss']
    for i in range(1, 8):
        chart.add_row([i, i/20+0.75, 0.30-i/20])
    context.log_artifact(chart)


if __name__ == "__main__":
    context = get_or_create_ctx('train')
    p1 = context.get_param('p1', 1)
    p2 = context.get_param('p2', 'a-string')
    my_job(context, p1, p2)
```

The example **training.py** application can be invoked as a local task, as demonstrated in the following code from the MLRun [**mlrun_basics.ipynb**](examples/mlrun_basics.ipynb) example notebook:
```python
run = run_local(task, command='training.py')
```
Alternatively, you can invoke the application by using the `mlrun` CLI; edit the parameters, inputs, and/or secret information, as needed, and ensure that **training.py** is found in the execution path or edit the file path in the command:
```sh
mlrun run --name train -p p2=5 -i infile.txt=s3://my-bucket/infile.txt -s file=secrets.txt training.py
```

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="using-hyperparameters-for-job-scaling"></a>
### Using Hyperparameters for Job Scaling

Data science involves long computation times and data-intensive tasks.
To ensure efficiency and scalability, you need to implement parallelism whenever possible.
MLRun supports this by using two mechanisms:

1. Clustering &mdash; run the code on a distributed processing engine (such as Dask, Spark, or Horovod).
2. Load-balancing/partitioning &mdash; split (partition) the work across multiple workers.

MLRun functions and tasks can accept hyperparameters or parameter lists, deploy many parallel workers, and partition the work among the deployed workers.
The parallelism implementation is left to the runtime.
Each runtime may have its own method of concurrent tasks execution.
For example, the Nuclio serverless engine manages many micro threads in the same process, which can run multiple tasks in parallel.
In a containerized system like Kubernetes, you can launch multiple containers, each processing a different task.

MLRun supports parallelism.
For example, the following code demonstrates how to use hyperparameters to run the XGBoost model-training task from the example in the previous section (`xgb_train`) with different parameter combinations:
```python
    parameters = {
         "eta":       [0.05, 0.10, 0.20, 0.30],
         "max_depth": [3, 4, 5, 6, 8, 10],
         "gamma":     [0.0, 0.1, 0.2, 0.3],
         }

    task = NewTask(handler=xgb_train, out_path='/User/mlrun/data').with_hyper_params(parameters, 'max.accuracy')
    run = run_local(task)
```

This code demonstrates how to instruct MLRun to run the same task while choosing the parameters from multiple lists (grid search).
MLRun then records all the runs, but marks only the run with minimal loss as the selected result.
For parallelism, it would be better to use runtimes like Dask, Nuclio, or jobs.

Alternatively, you can run a similar task (with hyperparameters) by using the MLRun CLI (`mlrun`); ensure that **training.py** is found in the execution path or edit the file path in the command:
```sh
mlrun run --name train_hyper -x p1="[3,7,5]" -x p2="[5,2,9]" --out-path '/User/mlrun/data' training.py
```

You can also use a parameters file if you want to control the parameter combinations or if the parameters are more complex.
The following code from the example [**mlrun_basics.ipynb**](examples/mlrun_basics.ipynb) notebook demonstrates how to run a task that uses a CSV parameters file (**params.csv** in the current directory):
```python
    task = NewTask(handler=xgb_train).with_param_file('params.csv', 'max.accuracy')
    run = run_local(task)
```

> **Note:** Parameter lists can be used in various ways.
> For example, you can pass multiple parameter files and use multiple workers to process the files simultaneously instead of one at a time.

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="auto-code-deployment-n-containerization"></a>
### Automated Code Deployment and Containerization

MLRun adopts Nuclio serverless technologies for automatically packaging code and building containers.
This enables you to provide code with some package requirements and let MLRun build and deploy your software.

To build or deploy a function, all you need is to call the function's `deploy` method, which initiates a build or deployment job.
Deployment jobs can be incorporated in pipelines just like regular jobs (using the `deploy_step` method of the function or Kubernetes-job runtime), thus enabling full automation and CI/CD.

A functions can be built from source code or from a function specification, web notebook, Git repo, or TAR archive.

A function can also be built by using the `mlrun` CLI and providing it with the path to a YAML function-configuration file.
You can generate such a file by using the `to_yaml` or `export` function method.
For example, the following CLI code builds a function from a **function.yaml** file in the current directory:
```sh
mlrun build function.yaml
```
Following is an example **function.yaml** configuration file:
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

For more examples of building and running functions remotely using the MLRun CLI, see the [**remote**](examples/remote.md) example.

You can also convert your web notebook to a containerized job, as demonstrated in the following sample code; for a similar example with more details, see the [**mlrun_jobs.ipynb**](examples/mlrun_jobs.ipynb) example:

```python
# Create an ML function from the notebook code and annotations, and attach a
# v3io Iguazio Data Science Platform data volume to the function
fn = code_to_function(kind='job').apply(mount_v3io())

# Prepare an image from the dependencies to allow updating the code and
# parameters per run without the need to build a new image
fn.build(image='mlrun/nuctest:latest')
```

[Back to top](#top)

<a id="run-ml-workflow-w-kubeflow-pipelines"></a>
### Running an ML Workflow with Kubeflow Pipelines

ML pipeline execution with MLRun is similar to CLI execution.
A pipeline is created by running an MLRun workflow.
MLRun automatically saves outputs and artifacts in a way that is visible to [Kubeflow Pipelines](https://github.com/kubeflow/pipelines), and allows interconnecting steps.

For an example of a full ML pipeline that's implemented in a web notebook, see the XGBoost MLRun demo ([**demo-xgb-project**](https://github.com/mlrun/demo-xgb-project)).
The  [**train-xgboost.ipynb**](https://github.com/mlrun/demo-xgb-project/blob/master/notebooks/train-xgboost.ipynb) demo notebook includes the following code for implementing an XGBoost ML-training pipeline:
```python
@dsl.pipeline(
    name='My XGBoost training pipeline',
    description='Demonstrates how to use MLRun.'
)
def xgb_pipeline(
   eta = [0.1, 0.2, 0.3], gamma = [0.1, 0.2, 0.3]
):

    ingest = xgbfn.as_step(name='ingest_iris', handler='iris_generator',
                          outputs=['iris_dataset'])


    train = xgbfn.as_step(name='xgb_train', handler='xgb_train',
                          hyperparams = {'eta': eta, 'gamma': gamma},
                          selector='max.accuracy',
                          inputs = {'dataset': ingest.outputs['iris_dataset']},
                          outputs=['model'])


    plot = xgbfn.as_step(name='plot', handler='plot_iter',
                         inputs={'iterations': train.outputs['iteration_results']},
                         outputs=['iris_dataset'])

    # Deploy the model-serving function with inputs from the training stage
    deploy = srvfn.deploy_step(project = 'iris', models={'iris_v1': train.outputs['model']})
```

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="db-operations"></a>
### Viewing Run Data and Performing Database Operations

When you configure an MLRun database, the results, parameters, and input and output artifacts of each run are recorded in the database.
You can view the results and perform operations on the database by using either of the following methods:

- Using [the MLRun dashboard](#mlrun-ui)
- Using [DB methods](#mlrun-db-methods) from your code

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="mlrun-ui"></a>
#### The MLRun Dashboard

The MLRun dashboard is a graphical user interface (GUI) for working with MLRun and viewing run data.

<br><p align="center"><img src="mlrunui.png" width="800"/></p><br>

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="mlrun-db-methods"></a>
#### MLRun Database Methods

You can use the `get_run_db` DB method to get an MLRun DB object for a configured MLRun database or API service.
Then, use the DB object's `connect` method to connect to the database or API service, and use additional methods to perform different operations, such as listing run artifacts or deleting completed runs.
For more information and examples, see the [**mlrun_db.ipynb**](examples/mlrun_db.ipynb) example notebook, which includes the following sample DB method calls:
```python
from mlrun import get_run_db

# Get an MLRun DB object and connect to an MLRun database/API service.
# Specify the DB path (for example, './' for the current directory) or
# the API URL ('http://mlrun-api:8080' for the default configuration).
db = get_run_db('./').connect()

# List all runs
db.list_runs('').show()

# List all artifacts for version 'latest' (default)
db.list_artifacts('', tag='').show()

# Check different artifact versions
db.list_artifacts('ch', tag='*').show()

# Delete completed runs
db.del_runs(state='completed')
```

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="additional-info-n-examples"></a>
### Additional Information and Examples

- [Replacing Runtime Context Parameters from the CLI](#replace-runtime-context-param-from-cli)
- [Remote Execution](#remote-execution)
  - [Nuclio Example](#remote-execution-nuclio-example)
- [Running the MLRun Database/API Service](#run-mlrun-db-service)

<a id="replace-runtime-context-param-from-cli"></a>
#### Replacing Runtime Context Parameters from the CLI

You can use the MLRun CLI (`mlrun`) to run MLRun functions or code and change the parameter values.

For example, the following CLI command runs the example XGBoost training code from the previous tutorial examples:
```sh
python -m mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt training.py
```

When running this sample command, the CLI executes the code in the **training.py** application using the provided run information:
- The value of parameter `p1` is set to `5`, overwriting the current parameter value in the run context.
- The file **infile.txt** is downloaded from a remote "mybucket" AWS S3 bucket.
- The credentials for the S3 download are retrieved from a **secrets.txt** file in the current directory.

<a id="remote-execution"></a>
#### Remote Execution

You can also run the same MLRun code that you ran locally as a remote HTTP endpoint.

<a id="remote-execution-nuclio-example"></a>
##### Nuclio Example

For example, you can wrap the XGBoost training code from the previous tutorial examples within a serverless [Nuclio](https://nuclio.io) handler function, and execute the code remotely using a similar CLI command to the one that you used locally.

You can run the following code from a Jupyter Notebook to create a Nuclio function from the notebook code and annotations, and deploy the function to a remote cluster.

> **Note:**
> - Before running the code, install the [`nuclio-jupyter`](https://github.com/nuclio/nuclio-jupyter) package for using Nuclio from Jupyter Notebook.
> - The example uses `apply(mount_v3io()`to attach a v3io Iguazio Data Science Platform data-store volume to the function.
>   By default, the v3io mount mounts the home directory of the platform's running user into the `\\User` function path.

```python
# Create an `xgb_train` Nuclio function from the notebook code and annotations;
# add a v3io data volume and a multi-worker HTTP trigger for parallel execution
fn = code_to_function('xgb_train', runtime='nuclio:mlrun')
fn.apply(mount_v3io()).with_http(workers=32)

# Deploy the function
run = fn.run(task, handler='xgb_train')
```

To execute the code remotely, run the same CLI command as in the previous tutorial examples and just substitute the code file name at the end with your function's URL.
For example, run the following command and replace `<function endpoint>` with your remote function endpoint:
```sh
mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt http://<function-endpoint>
```

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="run-mlrun-service"></a>
### Running an MLRun Service

An MLRun service is a web service that manages an MLRun database for tracking and logging MLRun run information, and exposes an HTTP API for working with the database and performing MLRun operations.

You can create and run an MLRun service by using either of the following methods:
- [Using Docker](#run-mlrun-service-docker)
- [Using the MLRun CLI](#run-mlrun-service-cli)

> **Note:** For both methods, you can optionally configure the service port and/or directory path by setting the `MLRUN_httpdb__port` and `MLRUN_httpdb__dirpath` environment variables instead of the respective run parameters or CLI options.

<a id="run-mlrun-service-docker"></a>
#### Using Docker to Run an MLRun Service

Run the following command to use Docker to create and run an instance of the MLRun service; replace `<service-directory path>` with a path to the service directory:
```sh
docker run -p 8080:8080 -v <service-directory path>:/mlrun/db
```

<a id="run-mlrun-service-cli"></a>
#### Using the MLRun CLI to Run an MLRun Service

Use the `db` command of the MLRun CLI (`mlrun`) to create and run an instance of the MLRun service from the command line:
```sh
mlrun db [OPTIONS]
```

To see the supported options, run `mlrun db --help`:
```
Options:
  -p, --port INTEGER  HTTP port for serving the API
  -d, --dirpath TEXT  Path to the MLRun service directory
```

