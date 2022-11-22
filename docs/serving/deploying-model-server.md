(deploying-model-server)=
# Deploying a model server

Deploying models in MLRun uses a special function type `serving`. You can create a `serving` function using the `code_to_function()` call from a notebook. You can also import an existing serving function/template from the Function Hub.

This example converts a notebook to a serving function and adds a model to it:

```python
from mlrun import code_to_function
fn = code_to_function('my-function', kind='serving')
fn.add_model('m1', model_path=<model-artifact/dir>, class_name='MyClass', x=100)
``` 

See [`.add_model()`](../api/mlrun.runtimes.html#mlrun.runtimes.ServingRuntime.add_model) docstring for help and parameters.

See the full [Model Server example](https://github.com/mlrun/functions/blob/master/v2_model_server/v2_model_server.ipynb).

If you want to use multiple versions for the same model, use `:` to separate the name from the version. 
For example, if the name is `mymodel:v2` it means model name `mymodel` version `v2`.

You should specify the `model_path` (url of the model artifact/dir) and the `class_name` name 
(or class `module.submodule.class`). Alternatively, you can set the `model_url` for calling a 
model that is served by another function (can be used for ensembles).

The function object(fn) accepts many options. You can specify replicas range (auto-scaling), cpu/gpu/mem resources, add shared volume mounts, secrets, and any other Kubernetes resource through the `fn.spec` object or fn methods.

For example, `fn.gpu(1)` means each replica uses one GPU. 

To deploy a model, simply call:

```python
fn.deploy()
```

You can also deploy a model from within an ML pipeline (check the various demos for details).