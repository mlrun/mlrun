(available-steps)=
# Built-in steps

A step runs a function or class handler or a REST API call: MLRun comes with pre-built steps that include data manipulation, readers, writers and model serving.
All steps are supported by the storey engine. Support by any other engines is included in the step description, as relevant.

You can also write your own steps using standard Python functions or custom functions/classes, or a step can be an external REST API (the special `$remote` class).

See also [Data transformations](../feature-store/transformations.md#data-transformation-steps).

**In this section**
- [Base Operators](#base-operators)
- [External IO and data enrichment](#external-io-and-data-enrichment)
- [Models](#models)
- [Model serving](#model-serving)
- [Other](#other)

## Base Operators

| Class name                                                                                                                  | Description                                                                                                                                                                 |   
|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|      
|{py:class}`~storey.transformations.Batch`             | Batches events. This step emits a batch every `max_events` events, or when `timeout` seconds have passed since the first event in the batch was received.                   |
|{py:class}`~storey.transformations.Choice`           | Redirects each input element into one of the multiple downstreams.                                                                                                          |
|{py:class}`~storey.transformations.Extend`                       | Adds fields to each incoming event.                                                                                                                                         | 
|{py:class}`~storey.transformations.Filter`             | Filters events based on a user-provided function.                                                                                                                           | 
|{py:class}`~storey.transformations.FlatMap`          | Maps, or transforms, each incoming event into any number of events.                                                                                                         |
|{py:class}`~storey.transformations.Flatten`                    | Flatten is equivalent to FlatMap(lambda x: x).                                                                                                                              | 
|{py:class}`~storey.transformations.ForEach`          | Applies the given function on each event in the stream, and passes the original event downstream.                                                                           |
|{py:class}`~storey.transformations.MapClass`         | Similar to Map, but instead of a function argument, this class should be extended and its do() method overridden.                                                           |
|{py:class}`~storey.transformations.MapWithState` | Maps, or transforms, incoming events using a stateful user-provided function, and an initial state, which can be a database table.                                          |
|{py:class}`~storey.transformations.Partition`      | Partitions events by calling a predicate function on each event. Each processed event results in a Partitioned named tuple of (left=Optional[Event], right=Optional[Event]). |
|storey.Reduce                                       | Reduces incoming events into a single value that is returned upon the successful termination of the flow.                                                                   |
|{py:class}`~storey.transformations.SampleWindow` | Emits a single event in a window of `window_size` events, in accordance with `emit_period` and `emit_before_termination`.                                                   | 

## External IO and data enrichment
| Class name                                                                                                                    | Description                                                                                    |   
|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| {py:class}`~mlrun.serving.remote.BatchHttpRequests`                                                                           | A class for calling remote endpoints in parallel.                                              | 
| {py:class}`~mlrun.datastore.DataItem`                                                                                         | Data input/output class abstracting access to various local/remote data sources.               |
| {py:class}`~storey.transformations.JoinWithTable`                                                                   | Joins each event with data from the given table.                                               |
| JoinWithV3IOTable                                                                                                             | Joins each event with a V3IO table. Used for event augmentation.                               | 
| {py:class}`~storey.transformations.QueryByKey`                                 | Similar to AggregateByKey, but this step is for serving only and does not aggregate the event. | 
| {py:class}`~mlrun.serving.remote.RemoteStep`                                                                                  | Class for calling remote endpoints.                                                            | 
| {py:class}`~storey.transformations.SendToHttp`                                                   | Joins each event with data from any HTTP source. Used for event augmentation.                  |
 
## Models
| Class name                                                | Description                                                                                |   
|-----------------------------------------------------------|--------------------------------------------------------------------------------------------|
| {py:class}`~mlrun.frameworks.onnx.ONNXModelServer`        | A model serving class for serving ONYX Models. A sub-class of the  V2ModelServer class.    | 
| {py:class}`~mlrun.frameworks.pytorch.PyTorchModelServer`  | A model serving class for serving PyTorch Models. A sub-class of the  V2ModelServer class. |
| {py:class}`~mlrun.frameworks.sklearn.SKLearnModelServer`  | A model serving class for serving Sklearn Models. A sub-class of the  V2ModelServer class. |  
| {py:class}`~mlrun.frameworks.tf_keras.TFKerasModelServer` | A model serving class for serving TFKeras Models. A sub-class of the V2ModelServer class.  |
| {py:class}`~mlrun.frameworks.xgboost.XGBModelServer`      | A model serving class for serving XGB Models. A sub-class of the  V2ModelServer class.     | 

## Model serving

| Class name                                                  | Description                                                                                                                                                                                                                                                                   |        
|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| {py:class}`~mlrun.serving.routers.EnrichmentModelRouter`    | Auto enrich the request with data from the feature store. The router input accepts a list of inference requests (each request can be a dict or a list of incoming features/keys). It enriches the request with data from the specified feature vector (`feature_vector_uri`). |
| {py:class}`~mlrun.serving.routers.EnrichmentVotingEnsemble` | Auto enrich the request with data from the feature store. The router input accepts a list of inference requests (each request can be a dict or a list of incoming features/keys). It enriches the request with data from the specified feature vector (`feature_vector_uri`). |
| {py:class}`~mlrun.serving.routers.ModelRouter`              | Basic model router, for calling different models per each model path.                                                                                                                                                                                                         | 
| {py:class}`~mlrun.serving.routers.VotingEnsemble`           | An ensemble machine learning model that combines the prediction of several models.                                                                                                                                                                                            |     
| {py:class}`~mlrun.serving.ModelRunnerStep`                  | Runs multiple models on each event. When used in a graph, MLRun automatically imports the default language model class (LLModel) during function deployment. |

## Other
| Class name                                                 | Description                                                                                                   |   
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| {py:class}`~mlrun.feature_store.steps.FeaturesetValidator` | Validate feature values according to the feature set validation policy. Supported also by the Pandas engines. | 
| {py:class}`~mlrun.serving.steps.ChoiceByField`             | Routes events to downstream steps based on an event field that contains the step name or names. | 
| ReduceToDataFrame                                          | Builds a pandas DataFrame from events and returns that DataFrame on flow termination.                         |
