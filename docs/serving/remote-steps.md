(remote-steps)=
# Remote steps
Use remote steps to access HTTP sources, external URIs, and remote functions. 
This icon in the UI indicates remote steps: <img src="../_static/images/steps-remote.png" alt="graph-steps-remote" width="20"/>.

**In this section**
- [SendToHttp](#sendtohttp)
- [RemoteStep](#remotestep)
- [RemoteFunctionStep](#remotefunctionstep)

## SendToHttp

Joins each event with data from any HTTP source. Used for event augmentation. See {py:class}`~storey.transformations.SendToHttp`.

## RemoteStep

### Description

Use RemoteStep in both sync and async engines to invoke an external URI (HTTP or MLRun function). See {py:class}`~mlrun.serving.remote.RemoteStep`. 

### Examples
Using the `async` engine to trigger an external heavy process, such as a service generating a test using model and storing it in a DB would look similar to:
```Python
from mlrun.serving.remote import RemoteStep
flow = function.set_topology("flow", engine="async")
flow.to(name="step1", handler="func1").to(
    RemoteStep(name="remote_echo", url="<func-url>")).to(
    name="laststep", handler="func2").respond()
```

A typical example using the `sync` engine would be to get a prediction from a model:
```Python
flow = function.get_model_prediction("flow", engine="sync")
flow.to(name="step1", handler="func1").to(
    RemoteStep(name="prediction", url="<func-url>")).to(
    name="laststep", handler="func2").respond()
```

Example pipeline using an MLRun function URI, for example to get inference from a model:
```Python
flow = function.set_topology("flow", engine="async")
flow.to(name="step1", handler="func1").to(
    RemoteStep(name="prediction", url="<func-url>")).to(
    name="laststep", handler="func2").respond()
```

 
## RemoteFunctionStep
### Description
Calls a remote functions. See {py:class}`~mlrun.serving.remote.RemoteFunctionStep`.
### Use Case
Use this step when you want to invoke an **existing function deployed in MLRun** as part of a serving graph without manually specifying its HTTP endpoint.<br>
The step accepts a function name or URI, retrieves the function object from MLRun, and automatically resolves the function’s invocation URL.<br> This simplifies integration between serving graphs and previously deployed functions, especially when the endpoint address may change between environments.<br>
The remote function may belong to a different project. The function must expose an **HTTP trigger**.<br>
When the step executes, the incoming event is forwarded to the remote function via its resolved HTTP endpoint. The remote function response is forwarded to the next step in the graph.
### Example
    ```python
    # Reference an existing Nuclio function
    step = RemoteFunctionStep(fn="my-nuclio-function", project_name="my-project")

    # Create a serving function
    serving_fn = mlrun.new_function(name="serving-graph", kind="serving")

    # Build the serving graph
    graph = serving_fn.set_topology("flow")
    graph.to(step).respond()
    ```