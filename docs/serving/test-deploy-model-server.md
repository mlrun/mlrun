(test-deploy-model-server)=
# Test and deploy a model server

**In this section**
- [Testing the model](#testing-the-model)
- [Deploying the model](#deploying-the-model)

## Testing the model

MLRun provides a mock server as part of the `serving` runtime. This gives you the ability to deploy your serving function in your local environment for testing purposes.

```python
serving_fn = code_to_function(name='myService', kind='serving', image='mlrun/mlrun')
serving_fn.add_model('my_model', model_path=model_file_path)
server = serving_fn.to_mock_server()
```

You can use test data and programmatically invoke the `predict()` method of mock server. In this example, the model is expecting a python dictionary as input.

```python
my_data = '''{"inputs":[[5.1, 3.5, 1.4, 0.2],[7.7, 3.8, 6.7, 2.2]]}'''
server.test("/v2/models/my_model/infer", body=my_data)
```

<!-- Output:
2022-03-29 09:44:52,687 [info] model my_model was loaded
2022-03-29 09:44:52,688 [info] Loaded ['my_model'] 

    {'id': '0282c63bff0a44cabfb9f06c34489035',
    'model_name': 'my_model',
    'outputs': [0, 2]}
-->

The data structure used in the body parameter depends on how the `predict()` method of the model server is defined. For examples of how to define your own model server class, see [here](custom-model-serving-class.html#predict-method).

To review the mock server api, see [here](../api/mlrun.runtimes.html#mlrun.runtimes.ServingRuntime.to_mock_server).

## Deploying the model 

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