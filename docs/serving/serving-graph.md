# MLRun Serving Graphs

## Overview

MLRun serving graphs allow to easily build real-time data processing and 
advanced model serving pipelines, and deploy them quickly to production with
minimal effort.

The serving graphs can be composed of pre-defined graph blocks (model servers, routers,
ensembles, data readers and writers, data engineering tasks, validators, etc.), 
or from native python classes/functions. Graphs can auto-scale and span multiple function 
containers (connected through streaming protocols).

Graphs can run inside your IDE or Notebook for test and simulation and can be deployed 
into production serverless pipeline with a single command. Serving Graphs are built on 
top of [Nuclio](https://github.com/nuclio/nuclio) (real-time serverless engine), MLRun Jobs, 
[MLRun Storey](https://github.com/mlrun/storey) (native Python async and stream processing engine), 
and other MLRun facilities. 

### Accelerate performance and time to production
The underline Nuclio serverless engine uses high-performance parallel processing 
engine which maximize the utilization of CPUs and GPUs, support 13 protocols and 
invocation methods (HTTP, Cron, Kafka, Kinesis, ..), and dynamic auto-scaling for 
http and streaming. Nuclio and MLRun support the full life cycle, including auto 
generation of micro-services, APIs, load-balancing, logging, monitoring, and 
configuration management, allowing developers to focus on code, and deploy faster 
to production with minimal work.

### In this document

- [Examples](#examples)
  - [Simple model serving router](#simple-model-serving-router)
  - [Advanced data processing and serving ensemble](#advanced-data-processing-and-serving-ensemble)
  - [NLP processing pipeline with real-time streaming](#nlp-processing-pipeline-with-real-time-streaming)
- [The Graph State Machine](#the-graph-state-machine)
  - [Graph overview and usage](#graph-overview-and-usage)
  - [Graph context and Event objects](#graph-context-and-event-objects)
  - [Error handling and catchers](#error-handling-and-catchers)
  - [Implement your own task class or function](#implement-your-own-task-class-or-function)
  - [Building distributed graphs](#building-distributed-graphs)

## Examples

### Simple model serving router

in order to deploy a serving function you need to import or create the serving function, 
add models to it and deploy.  

```python
    import mlrun  
    # load the sklearn model serving function and add models to it  
    fn = mlrun.import_function('hub://v2_model_server')
    fn.add_model("model1", model_path={model1-url})
    fn.add_model("model2", model_path={model2-url})

    # deploy the function to the cluster
    fn.deploy()
    
    # test the live model endpoint
    fn.invoke('/v2/models/model1/infer', body={"inputs": [5]})
```

The Serving function support the same protocol used in KFServing V2 and Triton Serving framework,
In order to invoke the model you to use following url: `<function-host>/v2/models/model1/infer`.

See the [**serving protocol specification**](model-api.md) for details

> model url is either an MLRun model store object (starts with `store://`) or URL of a model directory 
(in NFS, s3, v3io, azure, .. e.g. s3://{bucket}/{model-dir}), note that credentials may need to 
be added to the serving function via environment variables or MLRun secrets

See the [**scikit-learn classifier example**](https://github.com/mlrun/functions/blob/master/sklearn_classifier/sklearn_classifier.ipynb) 
which explains how to create/log MLRun models.

#### **Writing your own serving class**

You can implement your own model serving or data processing classes, all you need to do is inherit the 
base model serving class and add your implementation for model `load()` (download the model file(s) and load the model into memory) 
and `predict()` (accept request payload and return prediction/inference results).

you can override additional methods : `preprocess`, `validate`, `postprocess`, `explain`,<br>
you can add custom api endpoint by adding method `op_xx(event)` (which can be invoked by
calling the `<model-url>/xx`, where operation = xx), see [model class API](./model-api.md).

#### **Minimal sklearn serving function example:**

> See the full [Model Server example](https://github.com/mlrun/functions/blob/master/v2_model_server/v2_model_server.ipynb).

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
import mlrun
from sklearn.datasets import load_iris

fn = mlrun.new_function('my_server', kind='serving')

# set the topology/router and add models
graph = fn.set_topology("router")
fn.add_model("model1", class_name="ClassifierModel", model_path="<path1>")
fn.add_model("model2", class_name="ClassifierModel", model_path="<path2>")

# create and use the graph simulator
server = fn.to_mock_server()
x = load_iris()['data'].tolist()
result = server.test("/v2/models/model1/infer", {"inputs": x})
```

### Advanced data processing and serving ensemble

MLRun Serving graphs can host advanced pipelines which handle event/data processing, ML functionality, 
 or any custom task, in the following example we build an asynchronous pipeline which pre-process data, 
pass the data into a model ensemble, and finishes off with post processing. 

**Check out the advanced [graph example notebook](graph-example.ipynb)**

create a new function of type serving from code and set the graph topology to `async flow`

```python
import mlrun
function = mlrun.code_to_function("advanced", filename="demo.py", 
                                  kind="serving", image="mlrun/mlrun",
                                  requirements=['storey'])
graph = function.set_topology("flow", engine="async")
```

Build and connect the graph (DAG) using the custom function and classes and plot the result. 
we add steps using the `step.to()` method (will add a new step after the current one), or using the 
`graph.add_step()` method.

We use the `graph.error_handler()` (apply to all steps) or `step.error_handler()` 
(apply to a specific step) if we want the error from the graph or the step to be 
fed into a specific step (catcher)

We can specify which step is the responder (returns the HTTP response) using the `step.respond()` method.If we don't specify the responder the graph will be non-blocking.

```python
# use built-in storey class or our custom Echo class to create and link Task steps
graph.to("storey.Extend", name="enrich", _fn='({"tag": "something"})') \
     .to(class_name="Echo", name="pre-process", some_arg='abc').error_handler("catcher")

# add an Ensemble router with two child models (routes), the "*" prefix mark it is a router class
router = graph.add_step("*mlrun.serving.VotingEnsemble", name="ensemble", after="pre-process")
router.add_route("m1", class_name="ClassifierModel", model_path=path1)
router.add_route("m2", class_name="ClassifierModel", model_path=path2)

# add the final step (after the router) which handles post processing and respond to the client
graph.add_step(class_name="Echo", name="final", after="ensemble").respond()

# add error handling step, run only when/if the "pre-process" step fail (keep after="")  
graph.add_step(handler="error_catcher", name="catcher", full_event=True, after="")

# plot the graph (using Graphviz) and run a test
graph.plot(rankdir='LR')
```

<br><img src="../_static/images/graph-flow.svg" alt="graph-flow" width="800"/><br>

create a mock (test) server, and run a test, you need to use `wait_for_completion()` 
to wait for the async event loop to complete
  
```python
server = function.to_mock_server()
resp = server.test("/v2/models/m2/infer", body={"inputs": data})
server.wait_for_completion()
``` 

and finally, you can deploy the graph as a real-time Nuclio serverless function with one command:

    function.deploy()

```{note}
If you test a Nuclio function that has a serving graph with the async engine via the Nuclio UI, the UI may not display the logs in the output.
```

### NLP processing pipeline with real-time streaming 

In Some cases we want to split our processing to multiple functions and use 
streaming protocols to connect those functions, in the following example we do the data 
processing in the first function/container and the NLP processing in the second function 
(for example if we need a GPU just for that part).

See the [full notebook example](./distributed-graph.ipynb)

```python
# define a new real-time serving function (from code) with an async graph
fn = mlrun.code_to_function("multi-func", filename="./data_prep.py", kind="serving", image='mlrun/mlrun')
graph = fn.set_topology("flow", engine="async")

# define the graph steps (DAG)
graph.to(name="load_url", handler="load_url")\
     .to(name="to_paragraphs", handler="to_paragraphs")\
     .to("storey.FlatMap", "flatten_paragraphs", _fn="(event)")\
     .to(">>", "q1", path=internal_stream)\
     .to(name="nlp", class_name="ApplyNLP", function="enrich")\
     .to(name="extract_entities", handler="extract_entities", function="enrich")\
     .to(name="enrich_entities", handler="enrich_entities", function="enrich")\
     .to("storey.FlatMap", "flatten_entities", _fn="(event)", function="enrich")\
     .to(name="printer", handler="myprint", function="enrich")\
     .to(">>", "output_stream", path=out_stream)

# specify the "enrich" child function, add extra package requirements
child = fn.add_child_function('enrich', './nlp.py', 'mlrun/mlrun')
child.spec.build.commands = ["python -m pip install spacy",
                             "python -m spacy download en_core_web_sm"]
graph.plot()
```

> Currently queues only support iguazio v3io stream, Kafka support will soon be added 

## The Graph State Machine

### Graph overview and usage

MLRun Graphs enable building and running DAGs (directed acyclic graph), the first graph element accepts 
an `Event` object, transform/process the event and pass the result to the next steps 
in the graph. The final result can be written out to some destination (file, DB, stream, ..)
or return back to the caller (one of the graph steps can be marked with `.respond()`).

The graph can host 4 types of steps:

* **Task** – simple execution step which follow other steps and runs a function or class handler or a 
  REST API call, tasks use one of many pre-built operators, readers and writers, can be standard Python 
  functions or custom functions/classes, or can be a external REST API (the special `$remote` class).  
* **Router** – emulate a smart router with routing logic and multiple child routes/models 
  (each is a tasks), the basic routing logic is to route to the child routes based on the Event.path, 
  more advanced or custom routing can be used, for example the Ensemble router sends the event to all 
  child routes in parallel, aggregate the result and respond (see the example). 
* **Queue** – queue or stream which accept data from one or more source steps and publish 
  to one or more output steps, queues are best used to connect independent functions/containers.
  queue can run in-memory or be implemented using a stream which allow it to span processes/containers. 
* **Flow** – A flow hosts the DAG with multiple connected tasks, routers or queues, it
  starts with some source (http request, stream, data reader, cron, etc.) and follow the 
  execution steps according to the graph layout, flow can have branches (in the async mode), 
  flow can produce results asynchronously (e.g. write to an output stream), or can respond synchronously 
  when one of the steps is marked as the responder (`step.respond()`).

The Graph server have two modes of operation (topologies): 
* **router topology** (default)- a minimal configuration with a single router and child tasks/routes, 
  this can be used for simple model serving or single hop configurations.
* **flow topology** - a full graph/DAG, the flow topology is implemented using two engines, `async` (the default)
  is based on [Storey](https://github.com/mlrun/storey) and async event loop, and `sync` which support a simple 
  sequence of steps.

Example for setting the topology:

    graph = function.set_topology("flow", engine="async")

### Graph context and Event objects

#### The Event object

The Graph state machine accepts an Event object (similar to Nuclio Event) and passes 
it along the pipeline, an Event object hosts the event `body` along with other attributes 
such as `path` (http request path), `method` (GET, POST, ..), `id` (unique event ID).

In some cases the events represent a record with a unique `key`, which can be read/set 
through the `event.key`, and records have associated `event.time` which by default will be 
the arrival time, but can also be set by a step.

The Task steps are called with the `event.body` by default, if a Task step need to 
read or set other event elements (key, path, time, ..) the user should set the task `full_event`
argument to `True`.

Task steps support optional `input_path` and `result_path` attributes which allow controlling which portion of 
the event will be sent as input to the step, and where to update the returned result.

for example, if we have an event body `{"req": {"body": "x"}}`, `input_path="req.body"` and `result_path="resp"` 
the step will get `"x"` as the input and the output after the step will be `{"req": {"body": "x"}: "resp": <step output>}`.
Note that `input_path` and `result_path` do not work together with `full_event=True`.

#### The Context object

the step classes are initialized with a `context` object (when they have `context` in their `__init__` args)
, the context is used to pass data and for interfacing with system services. The context object has the 
following attributes and methods.

Attributes:
* **logger** - central logger (Nuclio logger when running in Nuclio)
* **verbose** - will be True if in verbose/debug mode
* **root** - the graph object
* **current_function** - when running in a distributed graph, the current child function name 

Methods:
* **get_param(key, default=None)** - get graph parameter by key, parameters are set at the
  serving function (e.g. `function.spec.parameters = {"param1": "x"}`)
* **get_secret(key)** - get the value of a project/user secret
* **get_store_resource(uri, use_cache=True)** - get mlrun store object (data item, artifact, model, feature set, feature vector)
* **get_remote_endpoint(name, external=False)** - return the remote nuclio/serving function http(s) endpoint given its [project/]function-name[:tag]
* **Response(headers=None, body=None, content_type=None, status_code=200)** - create nuclio response object, for returning detailed http responses

Example, using the context:

    if self.context.verbose:
        self.context.logger.info('my message', some_arg='text')
    x = self.context.get_param('x', 0)

### Error handling and catchers

Graph steps may raise an exception and we may want to have an error handling flow,
it is possible to specify exception handling step/branch which will be triggered on error,
the error handler step will receive the event which entered the failed step, with two extra
attributes: `event.origin_state` will indicate the name of the failed step, and `event.error`
will hold the error string.

We use the `graph.error_handler()` (apply to all steps) or `step.error_handler()` 
(apply to a specific step) if we want the error from the graph or the step to be 
fed into a specific step (catcher)

Example, setting an error catcher per step: 

    graph.add_step("MyClass", name="my-class", after="pre-process").error_handler("catcher")
    graph.add_step("ErrHandler", name="catcher", full_event=True, after="")
    
> Note: additional steps may follow our `catcher` step

see the full example [above](#advanced-data-processing-and-serving-ensemble)

**exception stream:**

The graph errors/exceptions can be pushed into a special error stream, this is very convenient 
in the case of distributed and production graphs 

setting the exception stream address (using v3io streams uri):

    function.spec.error_stream = 'users/admin/my-err-stream'
    

### Implement your own task class or function

The Graph executes built-in task classes or user provided classes and functions,
the task parameters include the following:
* `class_name` (str) - the relative or absolute class name
* `handler` (str) - the function handler (if class_name is not specified it is the function handler)
* `**class_args` - a set of class `__init__` arguments 

**Check out the [example notebook](graph-example.ipynb)**

you can use any python function by specifying the handler name (e.g. `handler=json.dumps`), 
the function will be triggered with the `event.body` as the first argument, and its result 
will be passed to the next step.

instead we can use classes which can also store some step/configuration and separate the 
one time init logic from the per event logic, the classes are initialized with the `class_args`,
if the class init args contain `context` or `name`, those will be initialize with the 
[graph context](#graph-context-and-event-objects) and the step name. 

the class_name and handler specify a class/function name in the `globals()` (i.e. this module) by default
or those can be full paths to the class (module.submodule.class), e.g. `storey.WriteToParquet`.
users can also pass the module as an argument to functions such as `function.to_mock_server(namespace=module)`,
in this case the class or handler names will also be searched in the provided module.

when using classes the class event handler will be invoked on every event with the `event.body` 
if the Task step `full_event` parameter is set to `True` the handler will be invoked and return
the full `event` object. If we don't specify the class event handler it will invoke the class `do()` method. 

if you need to implement async behavior you should subclass `storey.MapClass`.


### Building distributed graphs

Graphs can be hosted by a single function (using zero to N containers), or span multiple functions
where each function can have its own container image and resources (replicas, GPUs/CPUs, volumes, etc.).
it has a `root` function which is where you configure triggers (http, incoming stream, cron, ..), 
and optional downstream child functions.

Users can specify the `function` attribute in `Task` or `Router` steps, this will indicate where 
this step should run, when the `function` attribute is not specified it would run on the root function.
`function="*"` means the step can run in any of the child functions.

steps on different functions should be connected using a `Queue` step (a stream)

**adding a child function:**

```python
fn.add_child_function('enrich', 
                      './entity_extraction.ipynb', 
                      image='mlrun/mlrun',
                      requirements=["storey", "sklearn"])
```

see a [complete example](#nlp-processing-pipeline-with-real-time-streaming)  
