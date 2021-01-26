# MLRun - Serving

Mlrun serving can take MLRun models or standard model files and produce managed real-time serverless functions 
(based on Nuclio real-time serverless engine), which can be deployed everywhere. 

Simple model serving classes can be written in Python or be taken from a set of pre-developed 
ML/DL classes, the code can handle complex data, feature preparation, binary data (images/video).
The serving engine supports the full lifecycle including auto generation of micro-services, APIs, 
load-balancing, logging, model monitoring, configuration management, etc.  

The underline Nuclio serverless engine is built on top of a high-performance parallel processing engine 
which maximize the utilization of CPUs and GPUs, support 13 protocols 
and invocation methods (HTTP, Cron, Kafka, Kinesis, ..), and dynamic auto-scaling for http and streaming.

MLRun Serving V2 is using the same protocol as KFServing V2 and Triton Serving framework, 
but eliminate the operational overhead and provide additional functionality.

#### In This Document
- [Creating a Model](#creating-and-serving-a-model)
- [Writing a Simple Serving Class](#writing-a-simple-serving-class)
- [Models, Routers And Graphs](#models-routers-and-graphs)
- [Creating Model Serving Function (Service)](#creating-model-serving-function-service)
- [Model Server API](#model-server-api)
- [Model Monitoring](#model-monitoring)

<b>You can find [this notebook example](../../examples/v2_model_server.ipynb) with a V2 serving functions</b>

## Creating And Serving A Model

Models can be retrieved from MLRun model store or from various storage repositories
(local file, NFS, S3, Azure blob, Iguazio v3io, ..). When using the model store it packs additional 
information such ad metadata, parameters, metrics, extra files which can be used by the serving function.

When you run MLRun training job you can simply use `log_model()` (from within an MLRun function)
to store the model with its metadata:

```python
    context.log_model('my-model', body=bytes(xgb_model.save_raw()), 
                      model_file='model.pkl', 
                      metrics={"accuracy": 0.85}, parameters={'xx':'abc'},
                      labels={'framework': 'xgboost'})
```

You can use pre baked training functions which will take training data + params and produce the model + charts for you, 
[see example training function](https://github.com/mlrun/functions/blob/master/sklearn_classifier/sklearn_classifier.ipynb) for detailed usage.

Alternatively models can be specified by the URL of the model directory 
(in NFS, s3, v3io, azure, .. e.g. s3://{bucket}/{model-dir}), note that credentials may need to 
be added to the serving function via environment variables or secrets

serving the model:

in order to deploy a serving function you need to import or create the serving
function definition, add the model to it and deploy:

```python
    import mlrun  
    # load the sklearn model serving function and class  
    fn = mlrun.import_function('hub://v2_model_server')
    fn.add_model("mymodel", model_path={models-dir-url})
    fn.deploy()
```

## Writing A Simple Serving Class

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

**To test the function locally use the mock server:**

```python
from mlrun.serving.server import create_mock_server

models_path = '{model artifact/dir path}'
server = create_mock_server()
server.add_model("mymodel", class_name=ClassifierModel, model_path=models_path)

from sklearn.datasets import load_iris
x = load_iris()['data'].tolist()
result = server.test("mymodel/infer", {"inputs": x})
```

> Note: you can also create `mock_server` object from the function object 
using `fn.to_mock_server()`, this way the mock_server configuration will 
be identical to the function spec.

#### Load() method

in the load method we download the model from external store, run the algorithm/framework
`load()` call, and do any other initialization logic. 

load will run synchronously (the deploy will be stalled until load completes), 
this can be an issue for large models and cause a readiness timeout, we can increase the 
function `spec.readiness_timeout`, or alternatively choose async loading (load () will 
run in the background) sy setting the function `spec.load_mode = "async"`.  

the function `self.get_model()` downloads the model metadata object and main file (into `model_file` path),
additional files can be accessed using the returned `extra_data` (dict of dataitem objects).

the model metadata object is stored in `self.model_spec` and provide model parameters, metrics, schema, etc.
parameters can be accessed using `self.get_param(key)`, the parameters can be specified in the model or during 
the function/model deployment.  

#### predict() method

the predict method is called when we access the `/infer` or `/predict` url suffix (operation).
the method accepts the request object (as dict), see [API doc](#model-server-api) below.
and should return the specified response object.

#### explain() method

the explain method provides a hook for model explanability, and is accessed using the `/explain` operation. .

#### pre/post and validate hooks

users can overwrite the `preprocess`, `validate`, `postprocess` methods for additional control 
The call flow is:

    pre-process -> validate -> predict/explain -> post-process 
    
## Models, Routers And Graphs

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

## Creating Model Serving Function (Service)

In order to provision a serving function we need to create an MLRun function of type `serving`
, this can be done by using the `code_to_function()` call from a notebook. We can also import 
an existing serving function/template from the marketplace.

Example (run inside a notebook): this code converts a notebook to a serving function and adding a model to it:

```python
from mlrun import code_to_function
fn = code_to_function('my-function', kind='serving')
fn.add_model('m1', model_path=<model-artifact/dir>, class_name='MyClass', parameters={"x": 100})
``` 

see `.add_model()` docstring for help and parameters, 
see [xgb_serving.ipynb](../../examples/xgb_serving.ipynb) notebook example.

If we want to use multiple versions for the same model, we use `:` to seperate the name from the version, 
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

to create a `mock server` for testing from the function spec use `fn.to_mock_server()`, example:

```python
server = fn.to_mock_server(globals())
result = server.test("/v2/models/mymodel/infer", {"inputs": x})
```

we can also deploy a model from within an ML pipeline (check the various demos for details).

## Model Server API

MLRun Serving follows the same REST API defined by Triton and [KFServing v2](https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md)

Nuclio also support streaming protocols (Kafka, kinesis, MQTT, ..), in the case of streaming 
the `model` name and `operation` can be encoded inside the message body

### get server info

GET / or /v2/health

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

* `stream_path` - the v3io stream path (e.g. `v3io:///users/..`)
* `sample` -  optional, sample every N requests
* `batch` -  optional, send micro-batches every N requests


