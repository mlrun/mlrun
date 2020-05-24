<a id="top"></a>
# Quick-Start <!-- omit in toc -->

- [Basic Components](#basic-components)
- [Managed and Portable Execution](#managed-and-portable-execution)
- [Automated Parameterization, Artifact Tracking, and Logging](#automated-parameterization-artifact-tracking-and-logging)
  - [Example](#example)
- [Using Hyperparameters for Job Scaling](#using-hyperparameters-for-job-scaling)
- [Automated Code Deployment and Containerization](#automated-code-deployment-and-containerization)
- [Running an ML Workflow with Kubeflow Pipelines](#running-an-ml-workflow-with-kubeflow-pipelines)
- [Viewing Run Data and Performing Database Operations](#viewing-run-data-and-performing-database-operations)
  - [The MLRun Dashboard](#the-mlrun-dashboard)
  - [MLRun Database Methods](#mlrun-database-methods)
- [Additional Information and Examples](#additional-information-and-examples)
  - [Replacing Runtime Context Parameters from the CLI](#replacing-runtime-context-parameters-from-the-cli)
  - [Remote Execution](#remote-execution)
    - [Nuclio Example](#nuclio-example)
- [Running an MLRun Service](#running-an-mlrun-service)
  - [Using the MLRun CLI to Run an MLRun Service](#using-the-mlrun-cli-to-run-an-mlrun-service)

<a id="basic-components"></a>
## Basic Components

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
## Managed and Portable Execution

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

[Back to top](#top)

<a id="auto-parameterization-artifact-tracking-n-logging"></a>
## Automated Parameterization, Artifact Tracking, and Logging

After running a job, you need to be able to track it, including viewing the run parameters, inputs, and outputs.
To support this, MLRun introduces a concept of a runtime **"context"**: the code can be set up to get parameters and inputs from the context, as well as log run outputs, artifacts, tags, and time-series metrics in the context.

<a id="auto-parameterization-artifact-tracking-n-logging-example"></a>
### Example

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
train_run = run_local(handler=xgb_train, pramas={'eta': 0.3})
```

Alternatively, you can replace the function with a serverless runtime to run the same code on a remote cluster, which could result in a ~10x performance boost.
You can find examples for different runtimes &mdash; such as a Kubernetes job, Nuclio, Dask, Spark, or an MPI job &mdash; in the MLRun [**examples**](examples) directory.

If you run your code from the `main` function, you can get the runtime context by calling the `get_or_create_ctx` method, as demonstrated in the following code from the MLRun [**training.py**](examples/training.py) example application.
The code also demonstrates how you can use the context object to read and write execution metadata, parameters, secrets, inputs, and outputs:

```python
from mlrun import get_or_create_ctx
from mlrun.artifacts import ChartArtifact, TableArtifact
import pandas as pd


def my_job(context, p1=1, p2='x'):
    # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)

    # get parameters from the runtime context (or use defaults)

    # access input metadata, values, files, and secrets (passwords)
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(context.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(context.get_input('infile.txt', 'infile.txt').get()))
    
    # Run some useful code e.g. ML training, data prep, etc.

    # log scalar result values (job result metrics)
    context.log_result('accuracy', p1 * 2)
    context.log_result('loss', p1 * 3)
    context.set_label('framework', 'sklearn')

    # log various types of artifacts (file, web page, table), will be versioned and visible in the UI
    context.log_artifact('model', body=b'abc is 123', local_path='model.txt', labels={'framework': 'xgboost'})
    context.log_artifact('html_result', body=b'<b> Some HTML <b>', local_path='result.html')
    context.log_artifact(TableArtifact('dataset', '1,2,3\n4,5,6\n', visible=True,
                                        header=['A', 'B', 'C']), local_path='dataset.csv')

    # create a chart output (will show in the pipelines UI)
    chart = ChartArtifact('chart')
    chart.labels = {'type': 'roc'}
    chart.header = ['Epoch', 'Accuracy', 'Loss']
    for i in range(1, 8):
        chart.add_row([i, i/20+0.75, 0.30-i/20])
    context.log_artifact(chart)

    raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
                'age': [42, 52, 36, 24, 73],
                'testScore': [25, 94, 57, 62, 70]}
    df = pd.DataFrame(raw_data, columns=[
        'first_name', 'last_name', 'age', 'testScore'])
    context.log_dataset('mydf', df=df, stats=True)


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

[Back to top](#top)

<a id="using-hyperparameters-for-job-scaling"></a>
## Using Hyperparameters for Job Scaling

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

[Back to top](#top)

<a id="auto-code-deployment-n-containerization"></a>
## Automated Code Deployment and Containerization

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
## Running an ML Workflow with Kubeflow Pipelines

ML pipeline execution with MLRun is similar to CLI execution.
A pipeline is created by running an MLRun workflow.
MLRun automatically saves outputs and artifacts in a way that is visible to [Kubeflow Pipelines](https://github.com/kubeflow/pipelines), and allows interconnecting steps.

For an example of a full ML pipeline that's implemented in a web notebook, see the Sklearn MLRun demo ([**demo-sklearn-project**](https://github.com/mlrun/demos/tree/master/sklearn-pipe)).
The  [**sklearn-project.ipynb**](https://github.com/mlrun/demos/tree/master/sklearn-pipe/sklearn-project.ipynb) demo notebook includes the following code for implementing an ML-training pipeline:
```python
from kfp import dsl
from mlrun import mount_v3io

funcs = {}
DATASET = 'iris_dataset'
LABELS  = "label"

def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        f.apply(mount_v3io())
        f.spec.image_pull_policy = 'Always'

@dsl.pipeline(
    name="My XGBoost training pipeline",
    description="Shows how to use mlrun."
)
def kfpipeline():
    
    # build our ingestion function (container image)
    builder = funcs['gen-iris'].deploy_step(skip_deployed=True)
    
    # run the ingestion function with the new image and params
    ingest = funcs['gen-iris'].as_step(
        name="get-data",
        handler='iris_generator',
        image=builder.outputs['image'],
        params={'format': 'pq'},
        outputs=[DATASET])

    # analyze our dataset
    describe = funcs["describe"].as_step(
        name="summary",
        params={"label_column": LABELS},
        inputs={"table": ingest.outputs[DATASET]})
    
    # train with hyper-paremeters 
    train = funcs["train"].as_step(
        name="train-skrf",
        params={"model_pkg_class" : "sklearn.ensemble.RandomForestClassifier",
                "sample"          : -1, 
                "label_column"    : LABELS,
                "test_size"       : 0.10},
        hyperparams={'CLASS_n_estimators': [100, 300, 500]},
        selector='max.accuracy',
        inputs={"dataset"         : ingest.outputs[DATASET]},
        outputs=['model', 'test_set'])

    # test and visualize our model
    test = funcs["test"].as_step(
        name="test",
        params={"label_column": LABELS},
        inputs={"models_path" : train.outputs['model'],
                "test_set"    : train.outputs['test_set']})

    # deploy our model as a serverless function
    deploy = funcs["serving"].deploy_step(models={f"{DATASET}_v1": train.outputs['model']})
```

[Back to top](#top)

<a id="db-operations"></a>
## Viewing Run Data and Performing Database Operations

When you configure an MLRun database, the results, parameters, and input and output artifacts of each run are recorded in the database.
You can view the results and perform operations on the database by using either of the following methods:

- Using [the MLRun dashboard](#mlrun-ui)
- Using [DB methods](#mlrun-db-methods) from your code

[Back to top](#top)

<a id="mlrun-ui"></a>
### The MLRun Dashboard

The MLRun dashboard is a graphical user interface (GUI) for working with MLRun and viewing run data.

<br><p align="center"><img src="_static/images/mlrunui.png" width="800"/></p><br>

[Back to top](#top)

<a id="mlrun-db-methods"></a>
### MLRun Database Methods

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

[Back to top](#top)

<a id="additional-info-n-examples"></a>
## Additional Information and Examples

- [Replacing Runtime Context Parameters from the CLI](#replace-runtime-context-param-from-cli)
- [Remote Execution](#remote-execution)
  - [Nuclio Example](#remote-execution-nuclio-example)
- [Running the MLRun Database/API Service](#run-mlrun-db-service)

<a id="replace-runtime-context-param-from-cli"></a>
### Replacing Runtime Context Parameters from the CLI

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
### Remote Execution

You can also run the same MLRun code that you ran locally as a remote HTTP endpoint.

<a id="remote-execution-nuclio-example"></a>
#### Nuclio Example

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

[Back to top](#top)

<a id="run-mlrun-service"></a>
## Running an MLRun Service

An MLRun service is a web service that manages an MLRun database for tracking and logging MLRun run information, and exposes an HTTP API for working with the database and performing MLRun operations.

You can create and run an MLRun service by using either of the following methods:
- [Using Docker](#run-mlrun-service-docker)
- [Using the MLRun CLI](#run-mlrun-service-cli)

> **Note:** For both methods, you can optionally configure the service port and/or directory path by setting the `MLRUN_httpdb__port` and `MLRUN_httpdb__dirpath` environment variables instead of the respective run parameters or CLI options.

<a id="run-mlrun-service-cli"></a>
### Using the MLRun CLI to Run an MLRun Service

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
