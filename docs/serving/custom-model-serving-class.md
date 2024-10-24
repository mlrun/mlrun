(custom-model-serving-class)=
# Build your own model serving class

Model serving classes implement the full model serving functionality, which includes
loading models, pre- and post-processing, prediction, explainability, and model monitoring.

Model serving classes must inherit from `mlrun.serving.V2ModelServer`, and at the minimum 
implement the `load()` (download the model file(s) and load the model into memory) 
and `predict()` (accept request payload and return prediction/inference results) methods.  

The class is initialized automatically by the model server and can run locally
as part of a Nuclio serverless function, or as part of a real-time pipeline.

You need to implement two mandatory methods:
  * **`load()`** &mdash; download the model file(s) and load the model into memory, 
  note this can be done synchronously or asynchronously.
  * **`predict()`** &mdash; accept request payload and return prediction/inference results.

You can override additional methods: `preprocess`, `validate`, `postprocess`, `explain`.  
Add a custom API endpoint by implementing the method `op_xx(event)`. Invoke it by
calling the `<model-url>/xx` (operation = `xx`).
    
**In this section**
* [Minimal sklearn serving function example](#minimal-sklearn-serving-function-example)
* [`load()` method](#load-method)
* [`predict()` method](#predict-method)
* [`explain()` method](#explain-method)
* [pre/post and validate hooks](#pre-post-and-validate-hooks)
* [Models, routers and graphs](#models-routers-and-graphs)
* [Creating a model serving function (service)](#creating-a-model-serving-function-service)
* [Model monitoring](#model-monitoring)
    
## Minimal sklearn serving function example

```python
from cloudpickle import load
import numpy as np
import mlrun


class ClassifierModel(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))

    def predict(self, body: dict) -> list:
        """Generate model predictions from sample"""
        feats = np.asarray(body["inputs"])
        result: np.ndarray = self.model.predict(feats)
        return result.tolist()
```
    
**Test the function locally using the mock server:**

```python
import mlrun
from sklearn.datasets import load_iris

project = mlrun.get_or_create_project("sklearn")
fn = project.set_function(name="my-server", kind="serving")

# set the topology/router and add models
graph = fn.set_topology("router")
fn.add_model("model1", class_name="ClassifierModel", model_path="<path1>")
fn.add_model("model2", class_name="ClassifierModel", model_path="<path2>")

# create and use the graph simulator
server = fn.to_mock_server()
x = load_iris()["data"].tolist()
result = server.test("/v2/models/model1/infer", {"inputs": x})
```

## `load()` method

In the load method, download the model from external store, run the algorithm/framework
`load()` call, and do any other initialization logic. 

The load runs synchronously (the deploy is stalled until load completes). 
This can be an issue for large models and cause a readiness timeout. You can increase the 
function `spec.readiness_timeout`, or alternatively choose async loading (where `load()` 
runs in the background) by setting the function `spec.load_mode = "async"`.  

The function `self.get_model()` downloads the model metadata object and main file (into `model_file` path).
Additional files can be accessed using the returned `extra_data` (dict of data-item objects).

The model metadata object is stored in `self.model_spec` and provides model parameters, metrics, schema, etc.
Parameters can be accessed using `self.get_param(key)`. The parameters can be specified in the model or during 
the function/model deployment.  

## `predict()` method

The predict method is called when you access the `/infer` or `/predict` URL suffix (operation).
The method accepts the request object (as dict), see [Model server API](model-api.html#infer-predict).
And it should return the specified response object.

## `explain()` method

The explain method provides a hook for model explainability, and is accessed using the `/explain` operation.

## pre/post and validate hooks

You can overwrite the `preprocess`, `validate`, and `postprocess` methods for additional control 
The call flow is:

```{mermaid}
flowchart LR
    id1(pre-process) --> id2(validate) --> id3(predict/explain) --> id4(post-process)
```
    
## Models, routers and graphs

Every serving function can host multiple models and logical steps. Multiple functions 
can connect in a graph to form complex real-time pipelines.

The basic serving function has a logical `router` with routes to multiple child `models`. 
The URL or the message determines which model is selected, e.g. using the URL schema:

    /v2/models/<model>[/versions/<ver>]/operation

```{admonition} Note
The `model`, `version` and `operation` can also be specified in the message body 
to support streaming protocols (e.g. Kafka).
```
       
More complex routers can be used to support ensembles (send the request to all child models 
and aggregate the result), multi-armed-bandit, etc. 

You can use a pre-defined Router class, or write your own custom router. 
Routers can route to models on the same function or access models on a separate function.

To specify the topology, router class and class arguments use `.set_topology()` with your function.

## Creating a model serving function (service)

To provision a serving function, you need to create an MLRun function of type `serving`.
This can be done by using the `code_to_function()` call from a notebook. You can also import 
an existing serving function/template from the Function Hub.

Example (run inside a notebook): this code converts a notebook to a serving function and adding a model to it:

```python
from mlrun import code_to_function

fn = code_to_function("my-function", kind="serving")
fn.add_model("m1", model_path="<model-artifact/dir>", class_name="MyClass", x=100)
``` 

See [`.add_model()`](../api/mlrun.runtimes.html#mlrun.runtimes.ServingRuntime.add_model) docstring for help and parameters.

See the full [Model Server example](https://github.com/mlrun/functions/blob/master/v2_model_server/v2_model_server.ipynb).

If you want to use multiple versions for the same model, use `:` to separate the name from the version. 
For example, if the name is `mymodel:v2` it means model name `mymodel` version `v2`.

You should specify the `model_path` (URL of the model artifact/dir) and the `class_name`
(or class `module.submodule.class`). Alternatively, you can set the `model_url` for calling a 
model that is served by another function (can be used for ensembles).

The function object(`fn`) accepts many options. You can specify replicas range (auto-scaling), cpu/gpu/mem resources, add shared 
volume mounts, secrets, and any other Kubernetes resource through the `fn.spec` object or function methods.

For example, `fn.gpu(1)` means each replica uses one GPU. 

To deploy a model, simply call:

```python
fn.deploy()
```

You can also deploy a model from within an ML pipeline (check the various demos for details).

## Model monitoring

Model activities can be tracked into a real-time stream and time-series DB. The monitoring data
is used to create real-time dashboards, detect drift, and analyze performance. 

To monitor a deployed model, apply `set_tracking()` on your serving function and specify the function spec attributes:

```py
fn.set_tracking(stream_path, batch, sample)
```

Optional arguments:
* **stream_path** &mdash; Enterprise: the v3io stream path (e.g. `v3io:///users/..`); CE: a valid Kafka stream 
(e.g. kafka://kafka.default.svc.cluster.local:9092)
* **sample** &mdash; optional, sample every N requests
* **batch** &mdash; optional, send micro-batches every N requests

Before you deploy a model with model monitoring enabled via `fn.set_tracking()`,
set the credentials for the project:

```py
project.set_model_monitoring_credentials(...)
```

See {ref}`model-monitoring-overview` for the full details.
