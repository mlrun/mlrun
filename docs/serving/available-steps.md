(basic-steps)=
# Basic steps



All steps are supported by the storey engine. 


**In this section**

- [Choice steps](#choice-steps)
- [Event operation steps](#event-operation-steps)
- [Batch operation steps](#batch-operation-steps)
- [Filter steps](#filter-steps)

## Choice steps
- [Choice](#choice)
- [ChoiceByField](#choicebyfield)

### Choice
- Description: Redirects each input element into one of the multiple downstreams. See {py:class}`~storey.transformations.Choice`.
- Use case:
- Example:

### ChoiceByField
- Description: Routes events to downstream steps based on an event field that contains the step name or names. See {py:class}`~mlrun.serving.steps.ChoiceByField`.
- Use case: Use this step when routing decisions in a serving graph should be determined dynamically based on a field in the event.
Instead of subclassing a choice step and implementing custom routing logic, you can add a field to the event containing the name (or names) of the downstream step(s) to route to.
The value of the configured field can be either:
    * a string – the event will be forwarded to the corresponding outlet.
    * a list or tuple of strings – the event will be forwarded to all specified outlets.

    This simplifies conditional routing logic by separating decision logic (a previous step that sets the field) from routing logic (handled by ChoiceByField).
- Example:
    ```
    # Create a serving function
    serving_fn = mlrun.new_function("choice-example", kind="serving")

    graph = serving_fn.set_topology("flow")

    # Step that decides the route and adds it to the event
    def choose_route(event):
        if isinstance(event["value"], dict):
            event["route"] = "dict"
        elif isinstance(event["value"],list):
            event["route"] = "list"
        else:
            raise AttributeError("Key 'route' in event must be either dict or list")
        return event

    def handle_dict(event):
        event["sum"] = sum(event["value"].values())
        return event

    def handle_list(event):
        event["sum"] = sum(event["value"])
        return event
        
    def pprint(event):
        print(f"sum is : {event['sum']}")
        return event

    graph.add_step(name="router", handler="choose_route")
    graph.add_step(class_name=ChoiceByField("route"), name="routing", after=["router"])
    graph.add_step(name="dict", handler="handle_dict", after=["routing"])
    graph.add_step(name="list", handler="handle_list", after=["routing"])
    graph.add_step(name="pprint", handler="pprint", after=["dict", "list"]).respond()
    ```


## Event operation steps 

- [DataItem](#dataitem)
- [Extend](#extend)
- [FlatMap](#flatmap)
- [Flatten](#flatten)
- [JoinWithTable](#joinwithtable)
- [JoinWithV3IOTable](#joinwithv3iotable)
- [MapClass](#mapclass)
- [MapWithState](#mapwithstate)
- [Partition](#partition)
- [Reduce](#storey-reduce)
- [SendToHttp](#sendtohttp)
- [ReduceToDataFrame](#reducetodataframe)

### DataItem
- Description: Data input/output class abstracting access to various local/remote data sources. See {py:class}`~mlrun.datastore.DataItem`.
- Use case:
- Example:

### Extend
- Description: Adds fields to each incoming event. See {py:class}`~storey.transformations.Extend`.
- Use case:
- Example:


### FlatMap
- Description: Maps, or transforms, each incoming event into any number of events. See {py:class}`~storey.transformations.FlatMap`.
- Use case:
- Example:


### Flatten 
- Description: Flatten is equivalent to FlatMap(lambda x: x). See {py:class}`~storey.transformations.Flatten`.
- Use case:
- Example:


### JoinWithTable
- Description: Joins each event with data from the given table. See {py:class}`~storey.transformations.JoinWithTable` 
- Use case:
- Example:


### JoinWithV3IOTable 
- Description: Joins each event with a V3IO table. Used for event augmentation.  
- Use case:
- Example:


### MapClass
- Description: Similar to Map, but instead of a function argument, this class should be extended and its do() method overridden. See {py:class}`~storey.transformations.MapClass`.
- Use case:
- Example:


### MapWithState
- Description: Maps, or transforms, incoming events using a stateful user-provided function, and an initial state, which can be a database table. See {py:class}`~storey.transformations.MapWithState`.
- Use case:
- Example:


### Partition
- Description: Partitions events by calling a predicate function on each event. Each processed event results in a Partitioned named tuple of (left=Optional[Event], right=Optional[Event]). See {py:class}`~storey.transformations.Partition` .
- Use case:
- Example:

(storey-reduce)=
### storey.Reduce
- Description: Reduces incoming events into a single value that is returned upon the successful termination of the flow. 
- Use case:
- Example:





### SendToHttp
- Description: Joins each event with data from any HTTP source. Used for event augmentation. See {py:class}`~storey.transformations.SendToHttp`.
- Use case:
- Example:


### ReduceToDataFrame 
- Description:  Builds a pandas DataFrame from events and returns that DataFrame on flow termination. 
- Use case:
- Example:





## Batch operation steps 
- [Batch](#batch)
- [BatchHttpRequests](#batchhttprequests)
- [ForEach](#foreach)

### Batch
- Description: Batches events. This step emits a batch every `max_events` events, or when `timeout` seconds have passed since the first event in the batch was received. See {py:class}`~storey.transformations.Batch`. 
- Use Case: 
- Example:

### BatchHttpRequests
- Description: A class for calling remote step endpoints in parallel. See {py:class}`~mlrun.serving.remote.BatchHttpRequests`.
- Use Case: 
- Example:

### ForEach
- Description: Applies the given function on each event in the stream, and passes the original event downstream. See {py:class}`~storey.transformations.ForEach`. 
- Use Case: 
- Example:


## Filter steps
- [Filter](#filter)
- [SampleWindow](#samplewindow)

### Filter
- Description: Filters events based on a user-provided function. See {py:class}`~storey.transformations.Filter` .
- Use Case: 
- Example:

### SampleWindow
- Description: Emits a single event in a window of `window_size` events, in accordance with `emit_period` and `emit_before_termination`. See {py:class}`~storey.transformations.SampleWindow`.
- Use case:
- Example:


## Custom steps

- [VotingEnsemble](#votingensemble)
- [QueryByKey](#querybykey)
- [RemoteStep](#remotestep)
- [RemoteFunctionStep](#remotefunctionstep)
- [ONNXModelServer](#onnxmodelserver)
- [PyTorchModelServer](#pytorchmodelserver)
- [SKLearnModelServer](#sklearnmodelserver)
- [TFKerasModelServer](#tfkerasmodelserver)
- [XGBModelServer](#xgbmodelserver)
    
### VotingEnsemble
- Description: An ensemble machine learning model that combines the prediction of several models. See {py:class}`~mlrun.serving.routers.VotingEnsemble`.
- Use Case: 
- Example:

### QueryByKey 
- Description: Similar to AggregateByKey, but this step is for serving only and does not aggregate the event. See {py:class}`~storey.transformations.QueryByKey`.
- Use Case: 
- Example:

### RemoteStep
- Description: Calls remote endpoints. See {py:class}`~mlrun.serving.remote.RemoteStep`.
- Use Case: 
- Example:
 
### RemoteFunctionStep
- Description: Calls remote functions. See {py:class}`~mlrun.serving.remote.RemoteFunctionStep`.
- Use Case: Use this step when you want to invoke an **existing function deployed in MLRun** as part of a serving graph without manually specifying its HTTP endpoint.<br>
The step accepts a function name or URI, retrieves the function object from MLRun, and automatically resolves the function’s invocation URL.<br> This simplifies integration between serving graphs and previously deployed functions, especially when the endpoint address may change between environments.<br>
The remote function may belong to a different project. The function must expose an **HTTP trigger**.<br>
When the step executes, the incoming event is forwarded to the remote function via its resolved HTTP endpoint. The remote function response is forwarded to the next step in the graph.
- Example: 
    ```
    # Reference an existing Nuclio function
    step = RemoteFunctionStep(fn="my-nuclio-function", project_name="my-project")

    # Create a serving function
    serving_fn = mlrun.new_function(name="serving-graph", kind="serving")

    # Build the serving graph
    graph = serving_fn.set_topology("flow")
    graph.to(step).respond()
    ```

### ONNXModelServer
- Description: A model serving class for serving ONYX Models. A sub-class of the  V2ModelServer class. See {py:class}`~mlrun.frameworks.onnx.ONNXModelServer`.
- Use Case: 
- Example:

### PyTorchModelServer
- Description: A model serving class for serving PyTorch Models. A sub-class of the  V2ModelServer class. See {py:class}`~mlrun.frameworks.pytorch.PyTorchModelServer`.
- Use Case: 
- Example:

### SKLearnModelServer
- Description: A model serving class for serving Sklearn Models. A sub-class of the V2ModelServer class. See {py:class}`~mlrun.frameworks.sklearn.SKLearnModelServer`.
- Use Case: 
- Example:

### TFKerasModelServer 
- Description: A model serving class for serving TFKeras Models. A sub-class of the V2ModelServer class. See {py:class}`~mlrun.frameworks.tf_keras.TFKerasModelServer`.
- Use Case: 
- Example:

### XGBModelServer
- Description: See {py:class}`~mlrun.frameworks.xgboost.XGBModelServer`.
- Use Case: 
- Example:
