# MLrun
A generic an easy to use mechanism for data scientists and developers/engineers to describe and track code, metadata, 
inputs, and outputs of machine learning related tasks (executions).

Read more details in [this doc link](https://docs.google.com/document/d/1JRoWx4X7ld3fzQtdTGVIbcZx-5HzlYmkFiQz6ei8izE/edit?usp=sharing)

## General Concept and Motivation

A developer or data-scientist writes code in a local IDE or notebook, later on he would 
like to run the same code on a larger cluster using scale-out containers or functions, 
once the code is ready he may want to incorporate the code in an automated ML workflow 
(e.g. using KubeFlow Pipelines).

In the various `runtime` environments he would like to use different configurations, parameters, and data.
He would also like to record/version all the outputs and and associated inputs (lineage).
Data artifacts on a cluster usually come from remote data stores like S3, GCS, 
.. this means and extra layer of complexity to transfer data.

Wouldnt it be great if we could write the code once in simple `local` semantics, and have some layer automate the 
data movement, versioning, parameter substitution, outputs tracking, etc. ?

This is the goal for this package.

The code is in early development stages and provided as a reference, we would like to foster wide industry collaboration 
and the idea is to make all the resources pluggable, this way developers code to one API and can use various open source projects or commercial products.     

## Architecture

A user instantiates a `context` object using the `get_or_create_ctx` method, reading or writing metadata, secrets, inputs, 
or outputs is done through the context object. the context object can be used as is locally, 
in the same time context can be `injected` via the API, CLI, RPC, environment variables or other mechanisms.

checkout [training example](examples/training.py).

### Example Code

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
    print('file\n{}\n'.format(context.get_object('infile.txt').get()))
    
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

### Running the function inline or with a specific runtime

A user can invoke code through the `run_start` library function, see [examples notebook](examples/mlrun_games.ipynb)

```python
from mlrun import run_start
import yaml

# note: you need to create/specify a secrets file with credentials for remote data access (e.g. in S3 or v3io)
run_spec =  {'metadata':
                 {'labels': {
                     'owner': 'yaronh'}},
             'spec':
                 {'parameters': {'p1': 5}, 
                  'input_objects': [],
                  'log_level': 'info',
                  'secret_sources': [{'kind': 'file', 'source': 'secrets.txt'}],
                 }}

task = run_start(run_spec, command='example1.py', rundb='./')
print(yaml.dump(task)) 
```


user can select the runtime to use (inline code, sub process, dask, horovod, nuclio) through parameters in the `run_start` command, 
see the [examples notebook](examples/mlrun_games.ipynb) for details

### Using hyper parameters 

The same code can be run multiple times using different parameters per run, this can be done by simply setting the hyperparams attribute e.g.:

```python
# note: you need to create/specify a secrets file with credentials for remote data access (e.g. in S3 or v3io)
run_spec =  {'metadata':
                 {'labels': {
                     'owner': 'yaronh'}},
             'spec':
                 {'parameters': {'p1': 5}, 
                  'input_objects': [],
                  'log_level': 'info',
                  'secret_sources': [{'kind': 'file', 'source': 'secrets.txt'}],
                 }}

hyper = { 'p2': ['aa', 'bb', 'cc']}

task = run_start(run_spec, command='example1.py', rundb='./', hyperparams=hyper)
print(yaml.dump(task))
```

### Replacing Runtime Context Parameters form CLI

`python -m mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt training.py`

when running the command above:
* the parameter `p1` will be overwritten with `5`
* the file `infile.txt` will be loaded from a remote S3 bucket
* credentials (for S3 and the app) will be loaded from the `secrets.txt` file

### Running Against Remote Code/Function

The same code can be implemented as a remote HTTP endpoint e.g. using [nuclio serverless](https://github.com/nuclio/nuclio) functions

for example the same code can be wrapped in a nuclio handler and be remotely executed using the same CLI

#### Function Code

```python
from mlrun import get_or_create_ctx
import time


def handler(context, event):
    ctx = get_or_create_ctx('myfunc', event=event)
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    context.logger.info(
        f'Run: {ctx.name} uid={ctx.uid}:{ctx.iteration} Params: p1={p1}, p2={p2}')

    time.sleep(1)

    # log scalar values (KFP metrics)
    ctx.log_result('accuracy', p1 * 2)
    ctx.log_result('latency', p1 * 3)

    # log various types of artifacts (and set UI viewers)
    ctx.log_artifact('test.txt', body=b'abc is 123')
    ctx.log_artifact('test.html', body=b'<b> Some HTML <b>', viewer='web-app')

    context.logger.info('run complete!')
    return ctx.to_json()
```

#### Function Deployment

to deploy the function into a cluster you can run the following commands
(make sure you first installed the nuclio-jupyter package)

```python
import nuclio

spec = nuclio.ConfigSpec(cmd=['pip install git+https://github.com/v3io/mlrun.git'],
                         config={'spec.build.baseImage': 'python:3.6-jessie',
                                 'spec.triggers.web': {'kind': 'http', 'maxWorkers': 8}})

addr = nuclio.deploy_file('mycode.py',name='myfunc', project='mlrun', spec=spec)

```


> Note: add this repo to nuclio build commands (`pip install git+https://github.com/v3io/mlrun.git`)

To execute the code remotely just substitute the file name with the function URL

`python -m mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt http://<function-endpoint>`

### Running Inside a KubeFlow Pipeline

Running in a pipeline would be similar to running using the command line
mlrun will automatically save outputs and artifacts in a way which will be visible to KubeFlow, and allow interconnecting steps

see the [pipelines notebook example](examples/ml_pipe.ipynb)
```python
# run training using params p1 and p2, generate 2 registered outputs (model, dataset) to be listed in the pipeline UI
# user can specify the target path per output e.g. 'model.txt':'<some-path>', or leave blank to use the default out_path
def mlrun_train(p1, p2):
    return mlrun_op('training', 
                    command = this_path + '/training.py', 
                    params = {'p1':p1, 'p2':p2},
                    outputs = {'model.txt':'', 'dataset.csv':''},
                    out_path = artifacts_path,
                    rundb = db_path)
                    
# use data (model) from the first step as an input
def mlrun_validate(modelfile):
    return mlrun_op('validation', 
                    command = this_path + '/validation.py', 
                    inputs = {'model.txt':modelfile},
                    out_path = artifacts_path,
                    rundb = db_path)
```

You can use the function inside a DAG:

```python
@dsl.pipeline(
    name='My MLRUN pipeline',
    description='Shows how to use mlrun.'
)
def mlrun_pipeline(
   p1 = 5 , p2 = '"text"'
):
    # create a train step, apply v3io mount to it (will add the /User mount to the container)
    train = mlrun_train(p1, p2).apply(mount_v3io())
    
    # feed 1st step results into the secound step
    # Note: the '.' in model.txt must be substituted with '-'
    validate = mlrun_validate(train.outputs['model-txt']).apply(mount_v3io())
```

### Query The Run Results and Artifact Database

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

