# Mlrun Serving Graphs

## Overview

Mlrun serving graphs allow to easily build real-time data processing and 
advanced model serving pipelines, and deploy them quickly to production with
minimal effort.

The serving graphs can be composed of pre-defined graph blocks (model servers, routers,
ensembles, data readers and writers, data engineering tasks, validators, etc.), 
or from native python classes/functions. Graphs can auto-scale and span multiple function 
containers (connected through streaming protocols).


Graphs can run inside your IDE or Notebook for test and simulation and can be deployed 
into production serverless pipeline with a single command. Serving Graphs are built on 
top of [Nuclio](https://github.com/nuclio/nuclio) (real-time serverless engine), MLRun Jobs, [MLRun Storey](https://github.com/mlrun/storey) 
(native Python async and stream processing engine), and other MLRun facilities. 

### Accelerate performance and time to production
The underline Nuclio serverless engine uses high-performance parallel processing 
engine which maximize the utilization of CPUs and GPUs, support 13 protocols and 
invocation methods (HTTP, Cron, Kafka, Kinesis, ..), and dynamic auto-scaling for 
http and streaming. Nuclio and MLRun support the full life cycle, including auto 
generation of micro-services, APIs, load-balancing, logging, monitoring, and 
configuration management, allowing developers to focus on code, and deploy faster 
to production with minimal work.

### In this document

* [**Examples**](#examples)
    * [**Simple model serving router**](#simple-model-serving-router)
    * [**Advanced data processing and serving ensemble**](#advanced-data-processing-and-serving-ensemble)
    * [**NLP processing pipeline with real-time streaming**](#nlp-processing-pipeline-with-real-time-streaming)
* [**The Graph State Machine**](#the-graph-state-machine)
    * Graph serving modes and operation 
    * Router state 
    * Async Flows 
    * The Graph context and Event objects 
    * Error handling and catchers 
* Stateful stream processing and data-engineering 
* Serving function configuration, resources, and triggers 
* Using MLRun model repository with serving
* Model monitoring and drift analysis 
* Using class or function from other modules or files


## Examples

### Simple model serving router

in order to deploy a serving function you need to import or create the serving function, 
add models to it and deploy, you can read more about [advanced routing]() options.  

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

See the [**serving protocol specification**]() for details

> model url is either an MLRun model store object (starts with `store://`) or URL of a model directory 
(in NFS, s3, v3io, azure, .. e.g. s3://{bucket}/{model-dir}), note that credentials may need to 
be added to the serving function via environment variables or MLRun secrets

See the [**scikit-learn classifier example**](https://github.com/mlrun/functions/blob/master/sklearn_classifier/sklearn_classifier.ipynb) 
which explains how to create/log MLRun models.

#### **Writing your own serving class**

You can implement your own model serving or data processing classes, all you need is to inherit the 
base model serving class and add your implementation for model `load()` (download the model file(s) and load the model into memory) 
and `predict()` (accept request payload and return prediction/inference results).

you can override additional methods : `preprocess`, `validate`, `postprocess`, `explain`<br>
you can add custom api endpoint by adding method `op_xx(event)` (which can be invoked by
calling the `<model-url>/xx`, where operation = xx), see [model class API]().

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


create a new function of type serving and set the graph topology to `async flow`

```python
import mlrun
function = mlrun.new_function("advanced", kind="serving")
graph = function.set_topology("flow", engine="async")
```

add 3 steps one after the other using `.to()`:
* the first will run `json.loads()` on the event, in case of error it will jump to "catcher"
* the 2nd will apply a filter using `storey.Filter` class and a lambda function
* The 3rd use a custom pre processing class (and pass the some_arg='abc' to its init)

```python
graph.to(name="loads", handler="json.loads").error_handler("catcher") \
     .to("storey.Filter", name="filter", _fn='(event["type"]=="infer")') \
     .to(class_name="PreProcess", name="pre-process", some_arg='abc')

# add an Ensemble router with two child models (routes), the "*" prefix mark it is a router class
router = graph.add_step("*mlrun.serving.VotingEnsemble", name="ensemble", after="pre-process")
router.add_route("m1", class_name="ModelClass", model_path="{path1}")
router.add_route("m2", class_name="ModelClass", model_path="{path2}")

# add the final step (after the router) which handles post processing and respond to the client
graph.add_step(class_name="PostProcess", name="final", after="ensemble").respond()

# add error handling state, run only when/if the "loads" state fail (keep after="")  
graph.add_step(class_name="ErrorCatch", name="catcher", after="")

# plot the graph (using Graphviz) and run a test
graph.plot(rankdir='LR')
```
<br><img src="_static/images/graph-flow.png" alt="graph-flow" width="800"/><br>

create a mock (test) server, and run a test, you need to use `` to wait for 
the async event loop to complete
  
```python
server = function.to_mock_server()
resp = server.test("/v2/models/m2/infer", body={"inputs": [5]})
server.wait_for_completion()
``` 

The execution order in the graph is determined via the `after` parameter, we can add states 
using the `state.to()` method (will add a new state after the current one), or using the 
`graph.add_step()` method.

We can specify which state is the responder (returns the HTTP response) using the `.respond()` method,
if we dont specify the responder the graph will be non-blocking.

### NLP processing pipeline with real-time streaming 

In Some cases we want to split our processing to multiple functions and use 
streaming protocols to connect those functions, in the following example we do the data 
processing in the 1st function/container and the NLP processing in the 2nd function (for example if we need a GPU just for that part).

```python
import mlrun

# create a serving function from source code
fn = mlrun.code_to_function('nlp-pipeline', filename='./nlp1.py', kind='serving', image='mlrun/mlrun')

queue_stream = 'users/admin/mystream'
result_stream = 'users/admin/mystream'

graph = fn.set_topology("flow", engine="async", exist_ok=True)

# define the graph, indicate that some steps run on the "enrich" function
# use a queue (stream) between the 1st and the 2nd function, and write the results to a stream 
graph.to('URLDownloader')\
     .to('ToParagraphs')\
     .to('$queue', 'to_v3io', path=queue_stream)\
     .to("ApplyNLP", function='enrich')\
     .to('ExtractEntities', function='enrich')\
     .to('EnrichEntities', function='enrich')\
     .to('storey.Flatten', 'flatten',function='enrich')\
     .to('$queue', 'to_result', path=result_stream)

# specify the "enrich" child function, built from a notebook file
fn.add_child_function('enrich', './entity_extraction.ipynb', 'mlrun/mlrun')

# plot the graph
graph.plot(rankdir='LR')

# add v3io (stream) credentials/secrets to the function and deploy (will deploy both parent and child functions)
fn.apply(mlrun.v3io_cred())
fn.deploy()
```

> Currently queues only support iguazio v3io stream, Kafka support will soon be added 

## The Graph State Machine

MLRun Graphs enable building and running DAGs (directed acyclic graph), the first graph element accepts 
an `Event` object, transform/process the event and pass the result to the next states 
in the graph. The final result can be written out to some destination (file, stream, ..)
or return back to the caller (one of the graph states can be marked with `.respond()`).

The graph can host 4 types of states:

* **Task** – simple execution step which follow other steps and runs a function or class handler
* **Router** – emulate a smart router with routing logic and multiple child routes/models 
  (each is a tasks), the basic routing logic is to route to the child routes based on the Event.path, more advanced or custom routing can be used, for example the ensemble router sends the event to all child routes in parallel, aggregate the result and respond 
* **Queue** – queue or stream which accept data from one or more source states and publish 
  to one or more output states, queues are best used to connect independent functions/containers
* **Flow** – A flow hosts the DAG with multiple connected tasks, routers or queues, it
  starts with some source (http request, stream, data reader, cron, etc.) and follow the 
  execution steps according to the graph layout.
  
### Graph serving modes and operation


### Router state


### Async Flows


### The Graph context and Event objects


### Error handling and catchers

