# How realtime pipelines work

## Graph overview and usage

MLRun graphs enable building and running DAGs (directed acyclic graph). The first graph element accepts 
an `Event` object, transforms/processes the event and passes the result to the next steps 
in the graph. The final result can be written out to some destination (file, DB, stream, ..)
or returned back to the caller (one of the graph steps can be marked with `.respond()`).

The graph can host 4 types of steps:

* **Task**: Simple execution step that follows other steps and runs a function, class handler, or a 
  REST API call. Tasks use one of many pre-built operators, readers and writers. They can be standard Python 
  functions or custom functions/classes, or an external REST API (the special `$remote` class).  
* **Router**: Emulates a smart router with routing logic and multiple child routes/models 
  (each is a task). The basic routing logic is to route to the child routes based on the Event.path.  
  More advanced or custom routing can be used, for example the Ensemble router sends the event to all 
  child routes in parallel, aggregates the result and responds (see the example). 
* **Queue**: Queues or streams that accept data from one or more source steps and publish 
  to one or more output steps. Queues are best used to connect independent functions/containers.
  A queue can run in-memory or be implemented using a stream such that it spans processes/containers. 
* **Flow**: A flow hosts the DAG with multiple connected tasks, routers or queues. It
  starts with a source (http request, stream, data reader, cron, etc.) and follows the 
  execution steps according to the graph layout. Flows can have branches (in the async mode), 
  and flows can produce results asynchronously (e.g. write to an output stream), or can respond synchronously 
  when one of the steps is marked as the responder (`step.respond()`).

The Graph server has two modes of operation (topologies): 
* **router topology** (default):A minimal configuration with a single router and child tasks/routes.  
  This topology can be used for simple model serving or single hop configurations.
* **flow topology**: A full graph/DAG: the flow topology is implemented using two engines, `async` (the default)
  is based on [Storey](https://github.com/mlrun/storey) and async event loop, and `sync` that supports a simple 
  sequence of steps.

Example for setting the topology:

    graph = function.set_topology("flow", engine="async")

## The Event object

The Graph state machine accepts an Event object (similar to Nuclio Event) and passes 
it along the pipeline. An Event object hosts the event `body` along with other attributes 
such as `path` (http request path), `method` (GET, POST, ..), and`id` (unique event ID).

In some cases the events represent a record with a unique `key`, which can be read/set 
through the `event.key`. Records have associated `event.time` that by default are 
the arrival time, but can also be set by a step.

The Task steps are called with the `event.body` by default. If a task step needs to 
read or set other event elements (key, path, time, ..) you should set the task `full_event`
argument to `True`.

Task steps support optional `input_path` and `result_path` attributes that allow controlling which portion of 
the event is sent as input to the step, and where to update the returned result.

For example, for an event body `{"req": {"body": "x"}}`, `input_path="req.body"` and `result_path="resp"` 
the step gets `"x"` as the input. The output after the step is `{"req": {"body": "x"}: "resp": <step output>}`.
Note that `input_path` and `result_path` do not work together with `full_event=True`.

## The Context object

The step classes are initialized with a `context` object (when they have `context` in their `__init__` args).
The context is used to pass data and for interfacing with system services. The context object has the 
following attributes and methods.

Attributes:
* **logger**: Central logger (Nuclio logger when running in Nuclio).
* **verbose**: True if in verbose/debug mode.
* **root**: The graph object.
* **current_function**: When running in a distributed graph, the current child function name.

Methods:
* **get_param(key, default=None)**: Get the graph parameter by key. Parameters are set at the
  serving function (e.g. `function.spec.parameters = {"param1": "x"}`).
* **get_secret(key)**: Get the value of a project/user secret.
* **get_store_resource(uri, use_cache=True)**: Get the mlrun store object (data item, artifact, model, feature set, feature vector).
* **get_remote_endpoint(name, external=False)**: Return the remote nuclio/serving function http(s) endpoint given its [project/]function-name[:tag].
* **Response(headers=None, body=None, content_type=None, status_code=200)**: Create a nuclio response object, for returning detailed http responses.

Example, using the context:

    if self.context.verbose:
        self.context.logger.info('my message', some_arg='text')
    x = self.context.get_param('x', 0)

## Error handling and catchers

Graph steps can raise an exception and you might want to have an error handling flow. ,
You can specify an exception handling step/branch that triggers on error.
The error handler step receives the event that entered the failed step, with two extra
attributes: `event.origin_state` indicates the name of the failed step, and `event.error`
holds the error string.

Use the `graph.error_handler()` (apply to all steps) or `step.error_handler()` 
(apply to a specific step) if you want the error from the graph or the step to be 
fed into a specific step (catcher).

Example, setting an error catcher per step: 

    graph.add_step("MyClass", name="my-class", after="pre-process").error_handler("catcher")
    graph.add_step("ErrHandler", name="catcher", full_event=True, after="")
    
```{note}
Additional steps can follow the `catcher` step.
```

See the full example [above](#advanced-data-processing-and-serving-ensemble)

**Exception stream:**

The graph errors/exceptions can be pushed into a special error stream. This is very convenient 
in the case of distributed and production graphs. 

Set the exception stream address (using v3io streams uri):

    function.spec.error_stream = 'users/admin/my-err-stream'
    

## Building distributed graphs

Graphs can be hosted by a single function (using zero to N containers), or span multiple functions
where each function can have its own container image and resources (replicas, GPUs/CPUs, volumes, etc.).
It has a `root` function, which is where you configure triggers (http, incoming stream, cron, ..), 
and optional downstream child functions.

You can specify the `function` attribute in `Task` or `Router` steps. This indicates where 
this step should run. When the `function` attribute is not specified it would run on the root function.
`function="*"` means the step can run in any of the child functions.

Steps on different functions should be connected using a `Queue` step (a stream)

**Adding a child function:**

```python
fn.add_child_function('enrich', 
                      './entity_extraction.ipynb', 
                      image='mlrun/mlrun',
                      requirements=["storey", "sklearn"])
```

See a [complete example](#nlp-processing-pipeline-with-real-time-streaming).  