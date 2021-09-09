# Model Serving API and Protocol

## Creating Custom Model Serving Class

Model serving classes implement the full model serving functionality which include
loading models, pre and post processing, prediction, explainability, and model monitoring.

Model serving classes must inherit from `mlrun.serving.V2ModelServer`, and at the minimum 
implement the `load()` (download the model file(s) and load the model into memory) 
and `predict()` (accept request payload and return prediction/inference results) methods.  

The class is initialized automatically by the model server and can run locally
as part of a nuclio serverless function, or as part of a real-time pipeline

You need to implement two mandatory methods:
  * **load()**     - download the model file(s) and load the model into memory, 
  note this can be done synchronously or asynchronously 
  * **predict()**  - accept request payload and return prediction/inference results

you can override additional methods : `preprocess`, `validate`, `postprocess`, `explain`<br>
you can add custom api endpoint by adding method `op_xx(event)`, it can be invoked by
calling the <model-url>/xx (operation = xx)

**minimal sklearn serving function example:**

```python
from cloudpickle import load
import numpy as np
import mlrun

class ClassifierModel(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model('.pkl')
        self.model = load(open(model_file, 'rb'))

    def predict(self, body: dict) -> list:
        """Generate model predictions from sample"""
        feats = np.asarray(body['inputs'])
        result: np.ndarray = self.model.predict(feats)
        return result.tolist()
```


### Load() method

in the load method we download the model from external store, run the algorithm/framework
`load()` call, and do any other initialization logic. 

load will run synchronously (the deploy will be stalled until load completes), 
this can be an issue for large models and cause a readiness timeout, we can increase the 
function `spec.readiness_timeout`, or alternatively choose async loading (load () will 
run in the background) by setting the function `spec.load_mode = "async"`.  

the function `self.get_model()` downloads the model metadata object and main file (into `model_file` path),
additional files can be accessed using the returned `extra_data` (dict of dataitem objects).

the model metadata object is stored in `self.model_spec` and provide model parameters, metrics, schema, etc.
parameters can be accessed using `self.get_param(key)`, the parameters can be specified in the model or during 
the function/model deployment.  

### predict() method

the predict method is called when we access the `/infer` or `/predict` url suffix (operation).
the method accepts the request object (as dict), see [API doc](#model-server-api) below.
and should return the specified response object.

### explain() method

the explain method provides a hook for model explainability, and is accessed using the `/explain` operation. .

### pre/post and validate hooks

users can overwrite the `preprocess`, `validate`, `postprocess` methods for additional control 
The call flow is:

    pre-process -> validate -> predict/explain -> post-process 
    
### Models, Routers And Graphs

Every serving function can host multiple models and logical steps, multiple functions 
can connect in a graph to form complex real-time pipelines.

The basic serving function has a logical `router` with routes to multiple child `models`, 
the url or the message will determine which model is selected, e.g. using the url schema:

    /v2/models/<model>[/versions/<ver>]/operation

> Note: the `model`, `version` and `operation` can also be specified in the message body 
to support streaming protocols (e.g. Kafka).

More complex routers can be used to support ensembles (send the request to all child models 
and aggregate the result), multi-armed-bandit, etc. 

You can use a pre-defined Router class, or write your own custom router. 
Router can route to models on the same function or access models on a separate function.

to specify the topology, router class and class args use `.set_topology()` with your function.

### Creating Model Serving Function (Service)

In order to provision a serving function we need to create an MLRun function of type `serving`
, this can be done by using the `code_to_function()` call from a notebook. We can also import 
an existing serving function/template from the marketplace.

Example (run inside a notebook): this code converts a notebook to a serving function and adding a model to it:

```python
from mlrun import code_to_function
fn = code_to_function('my-function', kind='serving')
fn.add_model('m1', model_path=<model-artifact/dir>, class_name='MyClass', x=100)
``` 

see `.add_model()` docstring for help and parameters

> See the full [Model Server example](https://github.com/mlrun/functions/blob/master/v2_model_server/v2_model_server.ipynb).

If we want to use multiple versions for the same model, we use `:` to separate the name from the version, 
e.g. if the name is `mymodel:v2` it means model name `mymodel` version `v2`.

User should specify the `model_path` (url of the model artifact/dir) and the `class_name` name 
(or class `module.submodule.class`), alternatively you can set the `model_url` for calling a 
model which is served by another function (can be used for ensembles).

the function object(fn) accepts many options, you can specify replicas range (auto-scaling), cpu/gpu/mem resources, add shared 
volume mounts, secrets, and any other Kubernetes resource through the `fn.spec` object or fn methods.

e.g. `fn.gpu(1)` means each replica will use one GPU.

to deploy a model we can simply call:

```python
fn.deploy()
```

we can also deploy a model from within an ML pipeline (check the various demos for details).

## Model Server API

MLRun Serving follows the same REST API defined by Triton and [KFServing v2](https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md)

Nuclio also support streaming protocols (Kafka, kinesis, MQTT, ..), in the case of streaming 
the `model` name and `operation` can be encoded inside the message body

### get server info

    GET /
    GET /v2/health

response example: `{'name': 'my-server', 'version': 'v2', 'extensions': []}`

### list models

    GET /v2/models/

response example:  `{"models": ["m1", "m2", "m3:v1", "m3:v2"]}`

### get model metadata

    GET v2/models/${MODEL_NAME}[/versions/${VERSION}]

response example: `{"name": "m3", "version": "v2", "inputs": [..], "outputs": [..]}`

### get model health / readiness

    GET v2/models/${MODEL_NAME}[/versions/${VERSION}]/ready

returns 200 for Ok, 40X for not ready.

### infer / predict

    POST /v2/models/<model>[/versions/{VERSION}]/infer

request body:

    {
      "id" : $string #optional,
      "model" : $string #optional
      "data_url" : $string #optional
      "parameters" : $parameters #optional,
      "inputs" : [ $request_input, ... ],
      "outputs" : [ $request_output, ... ] #optional
    }

- **id:** unique Id of the request, if not provided a random value will be provided
- **model:** model to select (for streaming protocols without URLs)
- **data_url:** option to load the `inputs` from an external file/s3/v3io/.. object
- **parameters:** optional request parameters
- **inputs:** list of input elements (numeric values, arrays, or dicts)
- **outputs:** optional, requested output values 

> Note: you can also send binary data to the function, for example a JPEG image, the serving engine pre-processor 
will detect that based on the HTTP content-type and convert it to the above request structure, placing the 
image bytes array in the `inputs` field.

response structure:

    {
      "model_name" : $string,
      "model_version" : $string #optional,
      "id" : $string,
      "outputs" : [ $response_output, ... ]
    }

### explain

POST /v2/models/<model>[/versions/{VERSION}]/explain

request body:

    {
      "id" : $string #optional,
      "model" : $string #optional
      "parameters" : $parameters #optional,
      "inputs" : [ $request_input, ... ],
      "outputs" : [ $request_output, ... ] #optional
    }

response structure:

    {
      "model_name" : $string,
      "model_version" : $string #optional,
      "id" : $string,
      "outputs" : [ $response_output, ... ]
    }

## Model Monitoring

Model activities can be tracked into a real-time stream and time-series DB, the monitoring data
used to create real-time dashboards and track model accuracy and drift. 
to set the streaming option specify the following function spec attributes:

to add tracking to your model add tracking parameters to your function:

    fn.set_tracking(stream_path, batch, sample)

* **stream_path** - the v3io stream path (e.g. `v3io:///users/..`)
* **sample** -  optional, sample every N requests
* **batch** -  optional, send micro-batches every N requests
