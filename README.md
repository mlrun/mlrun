# MLRun

[![CircleCI](https://circleci.com/gh/mlrun/mlrun/tree/development.svg?style=svg)](https://circleci.com/gh/mlrun/mlrun/tree/development)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version fury.io](https://badge.fury.io/py/mlrun.svg)](https://pypi.python.org/pypi/mlrun/)

A generic an easy to use mechanism for data scientists and developers/engineers 
to describe and run machine learning related tasks in various scalable runtime environments 
while automatically tracking code, metadata, inputs, and outputs of executions.
MLRun is integrated with [Nuclio serverless project](https://github.com/nuclio/nuclio).

Read more details in [this doc link](https://docs.google.com/document/d/1JRoWx4X7ld3fzQtdTGVIbcZx-5HzlYmkFiQz6ei8izE/edit?usp=sharing)

## General Concept and Motivation

A developer or data-scientist writes code in a local IDE or notebook, then he would 
like to run the same code on a larger cluster using scale-out containers or functions, 
once the code is ready he or another developer need to transfer the code into an automated ML workflow 
(e.g. using KubeFlow Pipelines), add logging, monitoring, security, etc. 

In the various (`runtime`) environments we use different configurations, parameters, and data sources.
We also use different frameworks and platforms which focus on different stages in the life-cycle.
This leads to constant development and DevOps/MLops work. 

When running experiments we would like to record/version all the outputs and and associated inputs 
(lineage), so we can easily reproduce or explain our results. The fact that we use different forms 
of storage (files, S3, ..) and databases doesnt make our life easy.

Wouldnt it be great if we could write the code once in simple `local` semantics and we can run it as is on various platforms.
imagine a layer automate the build process, execution, data movement, versioning, parametrization, outputs tracking, etc. 

This is the goal for this package.

The code is in early development stages and provided as a reference, we would like to foster wide industry collaboration 
and the idea is to make all the resources pluggable, this way developers code to one API and can use various open source projects or commercial products.     

## Content

<b>Architecture and tutorial</b>

* [Managed and portable execution a.k.a Serverless](#managed-and-portable-execution)
* [Automated parametrization, artifact tracking and logging](#automated-parametrization-artifact-tracking-and-logging)
* [Using hyper parameters for job scaling](#using-hyper-parameters-for-job-scaling)
* [Automated code deployment and containerization](#automated-code-deployment-and-containerization)
* [Running with KubeFlow ML Pipeline](#running-with-kubeflow-ml-pipeline)
* [Run and Artifact Database](#run-and-artifact-database)

<b>Examples & Notebooks</b>
* [Various run examples](examples/mlrun_games.ipynb)
* [From local runs to a scalable Kubernetes cluster](examples/nuclio_jobs.ipynb)
* [Automated workflows with KubeFlow Pipelines](examples/pipe_code.ipynb)
* [Using MLRUN with Kubernetes Job](examples/mlrun_jobs.ipynb)
* [Using MLRUN with Horovod and MpiJob](examples/mlrun_mpijob.ipynb)
* [Using MLRUN with Nuclio](examples/train_xgboost_serverless.ipynb)
* [Using MLRUN with Dask](examples/mlrun_dask.ipynb)
* [Using MLRUN with Spark - TBD]()
* [Query MLRUN DB](examples/mlrun_db.ipynb)
* [Automating container build](examples/build.py)

## Architecture

### Managed and portable execution

We have few main elements:

* task (run) - define the desired parameters, inputs, outputs and tracking of a run. 
Run can be created from a template and run over different `runtimes` or `functions`.
* function - a `runtime` specific software package and attributes (e.g. image, command, 
args, environment, ..). function can run one or many runs/tasks and can be created from templates.
* runtime - is a computation framework, we supports multiple `runtimes` such as local, 
kubernetes job, dask, nuclio, spark, mpijob (Horovod). runtimes may support 
parallelism and clustering (i.e. distribute the work among processes/containers).

example:

    task = NewTask(handler=handler, name='demo', params={'p1': 5})
    task.with_secrets('file', 'secrets.txt').set_label('type', 'demo')
    
    run = new_function(command='dask://').run(task)
    print(run.artifact('model'))

in this example the task defines our run spec (parameters, inputs, secrets, ..) .
we run this task on a `dask` function (local or clustered), and print out the result 
output (in this case the `model` artifact) or watch the progress of that run.

we can run the same `task` on different functions, enabling code portability and re-use, 
or we can use the same `function` to run different tasks or parameter combinations with 
minimal coding effort.

moving from run on a local notebook, to running in a container job, a scaled-out framework
or an automated workflow engine like KubeFlow is seamless, just swap the runtime/function.

### Automated parametrization, artifact tracking and logging 

Once our job run we need to track the run, their inputs, parameters and outputs.
`mlrun` introduces a concept of an ML `context`, the code can be instrumented to 
get parameters and inputs from the context as well as log outputs, artifacts, 
tag, and time-series metrics.


<b>Example, XGBoost training function</b>

```python
import xgboost as xgb
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

dtrain = dtest = Y_test = None

def load_dataset():
    global dtrain, dtest, Y_test
    iris = load_iris()
    y = iris['target']
    X = iris['data']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)


def xgb_train(context, model_name='iris_v1.bst',
            max_depth=6,
            num_class=10,
            eta=0.2,
            gamma=0.1,
            steps=20):
    global dtrain, dtest, Y_test

    if dtrain is None:
        load_dataset()

    # Get params from event
    param = {"max_depth": max_depth,
             "eta": eta, "nthread": 4,
             "num_class": num_class,
             "gamma": gamma,
             "objective": "multi:softprob"}

    # Train model
    xgb_model = xgb.train(param, dtrain, steps)

    preds = xgb_model.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    context.log_result('accuracy', float(accuracy_score(Y_test, best_preds)))

    os.makedirs('models', exist_ok=True)
    model_file = model_name #os.path.join('models', model_name)
    xgb_model.save_model(model_file)
    context.log_artifact('model', src_path=model_file, labels={'framework': 'xgboost'})
```  


The function above can be executed locally with parameters (p1, p2), the results and artifacts 
will be logged automatically into a database with a single command. 

    train_run = new_function().run(handler=xgb_train).with_params(eta=0.3)    

we can swap the `function` with a serverless runtime and the same will run on a cluster.<br>
this can result in 10x performance boost, see [this notebook](examples/train_xgboost_serverless.ipynb) for details.
more examples can be found in [`\examples`](examples) directory, with `kubernetes job`, `nuclio`, `dask`, or `mpijob` runtimes.
 
if we run our code from `main` we can get the runtime context by calling the `get_or_create_ctx`
method. 

The example below shows us how we can use the `context` object provide us various ways to
read and write metadata, secrets, inputs, or outputs.

<b>Example, obtaining and using the context</b>

```python
from mlrun import get_or_create_ctx
from mlrun.artifacts import ChartArtifact, TableArtifact


def my_job():
    # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)
    context = get_or_create_ctx('train')
    
    # get parameters from the runtime context (or use defaults)
    p1 = context.get_param('p1', 1)
    p2 = context.get_param('p2', 'a-string')

    # access input metadata, values, files, and secrets (passwords)
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(context.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(context.get_input('infile.txt').get()))
    
    # RUN some useful code e.g. ML training, data prep, etc.

    # log scalar result values (job result metrics)
    context.log_result('accuracy', p1 * 2)
    context.log_result('loss', p1 * 3)
    context.set_label('framework', 'sklearn')

    # log various types of artifacts (file, web page, table), will be versioned and visible in the UI
    context.log_artifact('model.txt', body=b'abc is 123', labels={'framework': 'xgboost'})
    context.log_artifact('results.html', body=b'<b> Some HTML <b>', viewer='web-app')
    context.log_artifact(TableArtifact('dataset.csv', '1,2,3\n4,5,6\n',
                                        viewer='table', header=['A', 'B', 'C']))

    # create a chart output (will show in the pipelines UI)
    chart = ChartArtifact('chart.html')
    chart.labels = {'type': 'roc'}
    chart.header = ['Epoch', 'Accuracy', 'Loss']
    for i in range(1, 8):
        chart.add_row([i, i/20+0.75, 0.30-i/20])
    context.log_artifact(chart)


if __name__ == "__main__":
    my_job()
```


the code above can be invoked by calling:

    run = new_function(command='training.py').run(task)

or using the cli (while substituting the parameter and input values):

    mlrun run --name train -p p2=5 -i infile.txt=s3://my-bocket/infile.txt -s file=secrets.txt training.py


### Using hyper parameters for job scaling

Data-science involve long-running compute and data intensive tasks, in order to gain 
efficiency we need to implement parallelism where ever we can, `mlrun` deliver scalability using two mechanisms:

1. Clustering - run the code on distributed processing engined (Dask, Spark, Horovod, ..)
2. Load-balancing/partitioning - partition the work to multiple workers 

mlrun can accept hyper-parameters or parameter lists, deploy many parallel workers, and partition the work among those.
the parallelism implementation is left to the `runtime`, each may have its own way to run tasks concurrently.
for example `nuclio` serverless engine manage many micro-threads in the same process which can run multiple tasks in parallel. 
In a containerized system like Kubernetes we can launch multiple containers each processing a different task.

In `mlrun` we implement parallelism you can run many parameter combinations for the `xgboost` 
function above by using `hyper params`:

    parameters = {
         "eta":       [0.05, 0.10, 0.20, 0.30],
         "max_depth": [3, 4, 5, 6, 8, 10],
         "gamma":     [0.0, 0.1, 0.2, 0.3],
         }

    task = NewTask(handler=xgb_train, out_path='/User/mlrun/data').with_hyper_params(parameters, 'max.accuracy')
    run = new_function().run(task)

    
The line above tells mlrun to run the same task while choosing the parameters from multiple lists (GridSearch).
it will record ALL the runs, but mark the one with minimal `loss` as the selected result.
for parallelism it would be freffered to use `runtimes` like `dask`, `nuclio`, or `jobs`.

This can also be done via the CLI:

    mlrun run --name train_hyper -x p1="[3,7,5]" -x p2="[5,2,9]" training.py

We can use a parameter file if we want to control the parameter combinations or if the parameters are more complex.

    task = NewTask(handler=xgb_train).with_param_file('params.csv', 'max.accuracy')
    run = new_function().run(task)

  
> Note: parameter list can be used for various tasks, another example is to pass a list of files and 
have multiple workers process them simultaneously instead of one at a time.


### Automated code deployment and containerization 

Mlrun adopts some of `nuclio` serverless technologies for automatically packaging code and building containers,
this way we can specify code with some package requirements and let the system build and deploy our software.

Examples:

```python
inline = """
print(1+1)
"""

build_image('repo/tests2:latest',
      requirements=['pandas'],
      inline_code=inline)
``` 

or

```python
build_image('yhaviv/ktests3:latest',
      source='git://github.com/hodisr/iguazio_example.git',
      base_image='python:3.6',
      commands=['python setup.py install'])
```

or this we can convert our notebook into a containerized job:

```python
# create an ML function from the notebook, attache it to iguazio data fabric (v3io)
fn = code_to_function(runtime='job').apply(mount_v3io())

# prepare an image from the dependencies, so we wont need to build the image every run 
fn.build(image='mlrun/nuctest:latest')
``` 

see [detailed example](examples/nuclio_jobs.ipynb)

## Running with KubeFlow ML Pipeline

Running in a pipeline would be similar to running using the command line
mlrun will automatically save outputs and artifacts in a way which will be visible to KubeFlow, and allow interconnecting steps

see the [pipelines notebook example](examples/pipe_code.ipynb)
```python
# create a job from the notebook, attache it to iguazio data fabric (v3io)
fn = code_to_function('training')

# builf the container image
fn.build(image='mlrun/xgb:latest') 

# define the pipeline
@dsl.pipeline(
    name='My XGBoost training pipeline',
    description='Shows how to use mlrun.'
)
def xgb_pipeline(
   eta = [0.1, 0.2, 0.3], gamma = 0.2
):
    run = NewTask(handler='xgb_train', out_path=artifacts_path, outputs=['model']).with_hyper_params({'eta': eta}, selector='max.accuracy').with_params(gamma=gamma)
    train = fn.with_code().to_step(run).apply(mount_v3io())


# run the pipeline
client = kfp.Client(namespace='default-tenant')
arguments = {'eta': [0.1, 0.2, 0.4]}
run_result = client.create_run_from_pipeline_func(xgb_pipeline, arguments, run_name='xgb 1', experiment_name='xgb')    
```

## Run and Artifact Database

if you have specified a `rundb` the results and artifacts from each run are recorded 

you can use various `db` methods, see the [example notebook](examples/mlrun_db.ipynb)

```python
from mlrun import get_run_db

# connect to a local file DB
db = get_run_db('./').connect()

# list all runs
db.list_runs('').show()

# list all artifact for version "latest"
db.list_artifacts('', tag='').show()

# check different artifact versions 
db.list_artifacts('ch', tag='*').show()

# delete completed runs
db.del_runs(state='completed')
```

## Additional Information and Examples

### Replacing Runtime Context Parameters form CLI

`python -m mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt training.py`

when running the command above:
* the parameter `p1` will be overwritten with `5`
* the file `infile.txt` will be loaded from a remote S3 bucket
* credentials (for S3 and the app) will be loaded from the `secrets.txt` file

### Running Against Remote Code/Function

The same code can be implemented as a remote HTTP endpoint e.g. using [nuclio serverless](https://github.com/nuclio/nuclio) functions

for example the same code can be wrapped in a nuclio handler and be remotely executed using the same CLI

#### Function Deployment

to deploy the function into a cluster you can run the following commands
(make sure you first installed the nuclio-jupyter package)

```python
# create the function from the notebook code + annotations, add volumes and parallel HTTP trigger
fn = code_to_function('xgb_train', runtime='nuclio')
fn.add_volume('User','~/').with_http(workers=32)

run = fn.run(task, handler='xgb_train')
```


To execute the code remotely just substitute the file name with the function URL

`python -m mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt http://<function-endpoint>`


### Running HTTP Database

#### Docker

Run with `docker run -p8080:8080 -v /path/to/db:/mlrun/db`


You can pass `MLRUN_httpdb__port` environment variable to change port.

#### Command Line

```
$ mlrun db --help
Usage: mlrun db [OPTIONS]

  Run HTTP database server

Options:
  -p, --port INTEGER  port to listen on
  -d, --dirpath TEXT  database directory (dirpath)
  --help              Show this message and exit.
```
