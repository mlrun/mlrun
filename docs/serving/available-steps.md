(basic-steps)=
# Basic steps



All steps are supported by the storey engine. 


**In this section**

- [Choice steps](#choice-steps)
- [Event operation steps](#event-operation-steps)
- [Batch operation steps](#batch-operation-steps)
- [Filter steps](#filter-steps)
- [Models](#models)

## Choice steps
- [Choice](#choice)
- [ChoiceByField](#choicebyfield)

This icon in the UI indicates choice steps: <img src="../_static/images/steps-choice.png" alt="graph-steps-choice" width="20"/>.
### Choice
- Description: Routes each event to one or more downstream branches based on custom logic. See {py:class}`~storey.transformations.Choice`.


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

This icon in the UI indicates event operation steps: <img src="../_static/images/steps-event-operation.png" alt="graph-steps-event-operation" width="20"/>.

| Class name  | Description                                                                                                |   
|-------------|------------------------------------------------------------------------------------------------------| 
|{py:class}`~storey.transformations.Collector`| Collects streaming chunks and emits a single event once all chunks for a stream are received. (It acts as a no-op passthrough for non-streaming events.)|    
|{py:class}`~storey.transformations.Extend` |Adds new fields to each event using values returned by a user-defined function.| 
|{py:class}`~storey.transformations.FlatMap`|Applies a function that can expand a single event into multiple downstream events.|
|{py:class}`~storey.transformations.Flatten` |Flattens iterable outputs so that each element is emitted as a separate event.| 
|{py:class}`~storey.transformations.JoinWithTable` |Joins each event with data from the given table. |
|JoinWithV3IOTable|Joins each event with a V3IO table. Used for event augmentation.  |
|{py:class}`~storey.transformations.MapClass`  | Similar to Map, but instead of a function argument, this class should be extended and its do() method overridden.|
|{py:class}`~storey.transformations.MapWithState` |Maps, or transforms, incoming events using a stateful user-provided function, and an initial state, which can be a database table.|
|{py:class}`~storey.transformations.Partition`      |Partitions events by calling a predicate function on each event. Routes each event to a left if condition is True or right branch if False.|
|storey.Reduce |Reduces incoming events into a single value that is returned upon the successful termination of the flow.|
|ReduceToDataFrame|Builds a pandas DataFrame from events and returns that DataFrame on flow termination. |

## Batch operation steps 

This icon in the UI indicates batch steps: <img src="../_static/images/steps-batch.png" alt="graph-steps-batch" width="20"/>.
| Class name  | Description                                                                                                |   
|-------------|------------------------------------------------------------------------------------------------------| 
|{py:class}`~storey.transformations.Batch`  |Collects events until the batch reaches a configured size or age, then sends them downstream together.|
|{py:class}`~mlrun.serving.remote.BatchHttpRequests`|Sends multiple HTTP requests to remote step endpoints in parallel for batch processing.|
|{py:class}`~storey.transformations.ForEach`|Runs custom logic for every event and then passes the original event downstream.|

## Filter steps 

This icon in the UI indicates filter steps: <img src="../_static/images/steps-filter.png" alt="graph-steps-filter" width="20"/>.
| Class name  | Description                                                                                                |   
|-------------|------------------------------------------------------------------------------------------------------| 
|{py:class}`~storey.transformations.Filter` |Filters events based on a user-provided function.      | 
|{py:class}`~storey.transformations.SampleWindow` |Samples a single event from every group of events based on a configured policy such as first or last.| 



## Model Server
| Class name  | Description                                                                         |   
|-----------------------------------------------------------|--------------------------------|
| {py:class}`~mlrun.frameworks.onnx.ONNXModelServer` | A model serving class for serving ONYX Models. A sub-class of the  V2ModelServer class. | 
| {py:class}`~mlrun.frameworks.pytorch.PyTorchModelServer`  | A model serving class for serving PyTorch Models. A sub-class of the  V2ModelServer class. |
| {py:class}`~mlrun.frameworks.sklearn.SKLearnModelServer`  | A model serving class for serving Sklearn Models. A sub-class of the  V2ModelServer class. |  
| {py:class}`~mlrun.frameworks.tf_keras.TFKerasModelServer` | A model serving class for serving TFKeras Models. A sub-class of the V2ModelServer class. |
| {py:class}`~mlrun.frameworks.xgboost.XGBModelServer` | A model serving class for serving XGB Models. A sub-class of the  V2ModelServer class. | 
