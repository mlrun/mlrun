(basic-steps)=
# Basic steps



All steps are supported by the storey engine. 


**In this section**

- [Choice](#choice)
- [Event operations](#event-operations)
- [Batch operations](#batch-operations)
- [Filter](#filter)



## Choice

| Class name      | Description                                                                                                                        |   
|-----------------|------------------------------------------------------------------------------------------------------------------------------------|     
|{py:class}`~storey.transformations.Choice`           | Redirects each input element into one of the multiple downstreams.                 |
|{py:class}`~mlrun.serving.steps.ChoiceByField`             | Routes events to downstream steps based on an event field that contains the step name or names. | 


## Event operations 

| Class name      | Description                                                                                                                        |   
|-----------------|------------------------------------------------------------------------------------------------------------------------------------|      
|{py:class}`~mlrun.datastore.DataItem`               | Data input/output class abstracting access to various local/remote data sources.               |
|{py:class}`~storey.transformations.Extend`      | Adds fields to each incoming event.          | 
|{py:class}`~storey.transformations.FlatMap`          | Maps, or transforms, each incoming event into any number of events.   |
|{py:class}`~storey.transformations.Flatten`                    | Flatten is equivalent to FlatMap(lambda x: x).  | 
|{py:class}`~storey.transformations.JoinWithTable`        | Joins each event with data from the given table.              |
|JoinWithV3IOTable                                   | Joins each event with a V3IO table. Used for event augmentation.      |
|{py:class}`~storey.transformations.MapClass`         | Similar to Map, but instead of a function argument, this class should be extended and its do() method overridden.  |
|{py:class}`~storey.transformations.MapWithState` | Maps, or transforms, incoming events using a stateful user-provided function, and an initial state, which can be a database table.  |
|{py:class}`~storey.transformations.Partition`      | Partitions events by calling a predicate function on each event. Each processed event results in a Partitioned named tuple of (left=Optional[Event], right=Optional[Event]). |
|storey.Reduce   | Reduces incoming events into a single value that is returned upon the successful termination of the flow.         |
|{py:class}`~storey.transformations.SampleWindow` | Emits a single event in a window of `window_size` events, in accordance with `emit_period` and `emit_before_termination`.   | 
|{py:class}`~storey.transformations.SendToHttp`  | Joins each event with data from any HTTP source. Used for event augmentation.   |
|ReduceToDataFrame                                | Builds a pandas DataFrame from events and returns that DataFrame on flow termination.     |



## Batch operations 

| Class name      | Description                                                                                                                        |   
|-----------------|------------------------------------------------------------------------------------------------------------------------------------|   
|{py:class}`~storey.transformations.Batch`   | Batches events. This step emits a batch every `max_events` events, or when `timeout` seconds have passed since the first event in the batch was received.    |
|{py:class}`~mlrun.serving.remote.BatchHttpRequests`    | A class for calling remote step endpoints in parallel.     | 
|{py:class}`~storey.transformations.ForEach`   | Applies the given function on each event in the stream, and passes the original event downstream.      |


## Filter

| Class name      | Description                                                                                                                        |   
|-----------------|------------------------------------------------------------------------------------------------------------------------------------|   
|{py:class}`~storey.transformations.Filter`   | Filters events based on a user-provided function.      | 





## Custom

| Class name      | Description                                                                                                                        |   
|-----------------|------------------------------------------------------------------------------------------------------------------------------------|   
|{py:class}`~mlrun.serving.routers.VotingEnsemble`           | An ensemble machine learning model that combines the prediction of several models.  |     
|{py:class}`~mlrun.serving.ModelRunnerStep`                  | Runs multiple models on each event. When used in a graph, MLRun automatically imports the default language model class (LLModel) during function deployment. See [ModelRunnerStep](./model-serving-steps.md#modelrunnerstep).|
| {py:class}`~storey.transformations.QueryByKey`     | Similar to AggregateByKey, but this step is for serving only and does not aggregate the event. | 
| {py:class}`~mlrun.serving.remote.RemoteStep`        | Class for calling remote endpoints.        | 




 
## ????????
| Class name                                                | Description                                                                                |   
|-----------------------------------------------------------|--------------------------------------------------------------------------------------------|
| {py:class}`~mlrun.frameworks.onnx.ONNXModelServer`        | A model serving class for serving ONYX Models. A sub-class of the  V2ModelServer class.    | 
| {py:class}`~mlrun.frameworks.pytorch.PyTorchModelServer`  | A model serving class for serving PyTorch Models. A sub-class of the  V2ModelServer class. |
| {py:class}`~mlrun.frameworks.sklearn.SKLearnModelServer`  | A model serving class for serving Sklearn Models. A sub-class of the  V2ModelServer class. |  
| {py:class}`~mlrun.frameworks.tf_keras.TFKerasModelServer` | A model serving class for serving TFKeras Models. A sub-class of the V2ModelServer class.  |
| {py:class}`~mlrun.frameworks.xgboost.XGBModelServer`      | A model serving class for serving XGB Models. A sub-class of the  V2ModelServer class.     | 


