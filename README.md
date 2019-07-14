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

checkout [example1](example1.py) and [example2](example2.py).

### Example Code

```python
from mlrun import get_or_create_ctx
from mlrun.artifacts import TableArtifact, ChartArtifact

def my_job():
    # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)
    context = get_or_create_ctx('mytask')
    
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
    context.log_output('accuracy', p1 * 2)
    context.log_output('loss', p1 * 3)

    # log various types of artifacts (file, web page, table), will be versioned and visible in the UI
    context.log_artifact('model.txt', body=b'abc is 123')
    context.log_artifact('results.html', body=b'<b> Some HTML <b>', viewer='web-app')
    context.log_artifact(TableArtifact('dataset.csv', '1,2,3\n4,5,6\n',
                                        viewer='table', header=['A', 'B', 'C']))

    # create a chart output (will show in the pipelines UI)
    chart = ChartArtifact('chart.html')
    chart.header = ['Epoch','Accuracy', 'Loss']
    for i in range(1,8):
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

`python -m mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt example2.py`

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

def handler(context, event):
    ctx = get_or_create_ctx('myfunc', event=event)
    context.logger.info('This is an unstructured log')
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}')

    # log scalar values (KFP metrics)
    ctx.log_output('accuracy', p1 * 2)
    ctx.log_output('latency', p1 * 3)

    # log various types of artifacts (and set UI viewers)
    ctx.log_artifact('test.txt', body=b'abc is 123')
    ctx.log_artifact('test.html', body=b'<b> Some HTML <b>', viewer='web-app')

    return ctx.to_json()

```

> Note: add this repo to nuclio build commands (`pip install git+https://github.com/v3io/mlrun.git`)

To execute the code remotely just substitute the file name with the function URL

`python -m mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt http://<function-endpoint>`

### Running Inside a KubeFlow Pipeline

Running in a pipeline would be similar to running using the command line
mlrun will automatically save outputs and artifacts in a way which will be visible to KubeFlow, and allow interconnecting steps

```python
def mlrun_train(p1, p2):
    return mlrun_op('training', 
                    command = '/User/training.py', 
                    params = {'p1':p1, 'p2': p2},
                    outputs = {'model.txt':'', 'dataset.csv':''},
                    out_path ='v3io:///bigdata/mlrun/{{workflow.uid}}/',
                    rundb = '/User')
                    
# use data from the first step
def mlrun_validate(modelfile):
    return mlrun_op('validation', 
                    command = '/User/validation.py', 
                    inputs = {'model.txt':modelfile},
                    out_path ='v3io:///bigdata/mlrun/{{workflow.uid}}/',
                    rundb = '/User')
```

You can use the function inside a DAG:

```python
@dsl.pipeline(
    name='My MLRUN pipeline',
    description='Shows how to use mlrun.'
)
def mlrun_pipeline(
   p1 = 5, p2 = 'text'
):
    train = mlrun_train(p1, p2)
    
    # feed 1st step results into the secound step
    validate = mlrun_validate(train.outputs['model-txt'])
```

### Example Output

```yaml
metadata:
  name: mytask
  uid: eae6f665ff2c4ff3a212edd65645d08c
  project: ''
  tag: ''
  labels:
    owner: root
    workflow: 30b0b0a3-9ce4-11e9-b64f-0a581ce6bde8
  annotations: {}
spec:
  runtime:
    kind: ''
    command: /User/training.py
  parameters:
    p1: 5.5
    p2: another text
  input_objects:
  - key: infile.txt
    path: infile.txt
  data_stores: []
  output_artifacts:
  - key: model.txt
    path: ''
  - key: dataset.csv
    path: ''
  default_output_path: v3io:///bigdata/mlrun/30b0b0a3-9ce4-11e9-b64f-0a581ce6bde8/
status:
  state: running
  outputs:
    accuracy: 11.0
    loss: 16.5
  start_time: '2019-07-02 16:12:33.936275'
  last_update: '2019-07-02 16:12:33.936283'
  output_artifacts:
  - key: model.txt
    src_path: ''
    target_path: v3io:///bigdata/mlrun/30b0b0a3-9ce4-11e9-b64f-0a581ce6bde8/model.txt
    description: ''
    viewer: ''
  - key: results.html
    src_path: ''
    target_path: v3io:///bigdata/mlrun/30b0b0a3-9ce4-11e9-b64f-0a581ce6bde8/results.html
    description: ''
    viewer: web-app
  - key: dataset.csv
    src_path: ''
    target_path: v3io:///bigdata/mlrun/30b0b0a3-9ce4-11e9-b64f-0a581ce6bde8/dataset.csv
    description: ''
    format: ''
    header:
    - A
    - B
    - C
    viewer: table
  - key: chart.html
    src_path: ''
    target_path: v3io:///bigdata/mlrun/30b0b0a3-9ce4-11e9-b64f-0a581ce6bde8/chart.html
    description: ''
    viewer: chart
```