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
import json
import os
from mlrun import get_or_create_ctx

def my_func(ctx):
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(ctx.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(ctx.input_artifact('infile.txt').get()))

    ctx.log_output('accuracy', p1 * 2)
    for i in range(1,4):
        ctx.log_metric('loss', 2*i, i)
    ctx.log_artifact('chart.png')


if __name__ == "__main__":
    ex = get_or_create_ctx('mytask')
    my_func(ex)
    print(ex.to_yaml())
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
    print('accesskey = {}'.format(ctx.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(ctx.input_artifact('infile.txt').get()))

    ctx.log_output('accuracy', p1 * 2)
    for i in range(1,4):
        ctx.log_metric('loss', 2*i, i)
    ctx.log_artifact('chart.png')

    return ctx.to_yaml()
```

> Note: add this repo to nuclio build commands (`pip install git+https://github.com/v3io/mlrun.git`)

To execute the code remotely just substitute the file name with the function URL

`python -m mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt http://<function-endpoint>`

