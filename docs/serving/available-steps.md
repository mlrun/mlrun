# Built-in steps

MlRun provides you with many built-in stapes that you can use when building your graph. 

<!-- ##Using built-in steps -->

Click on the step names in the following sections to see the full usage.

- [Base Operators](#base-operators)
- [Data Transformations](#data-transformations)
- [External IO and data enrichment](#external-io-and-data-enrichment)
- [Sources](#sources)
- [Targets](#targets)
- [Models](#models)
- [Routers](#routers)
- [Other](#other)

## Base Operators

| Class name                                       | Description          |   
|--------------------------------------------------|---------------------------------------------------------------------------|      
| [storey.transformations.Batch](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.Batch) | Batches events. This step emits a batch every `max_events` events, or when `timeout` seconds have passed since the first event in the batch was received. | | 
| [storey.transformations.Choice](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.Choice) | Redirects each input element into one of the multiple downstreams.  |
| [storey.Extend](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.Extend) | Adds fields to each incoming event. | 
| [storey.transformations.Filter](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.Filter) | Filters events based on a user-provided function. | 
| [storey.transformations.FlatMap](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.FlatMap)  | Maps, or transforms, each incoming event into any number of events.  |
| [storey.steps.Flatten](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.Flatten)  |                              | 
| [storey.transformations.ForEach](https://storey.readthedocs.io/en/latest/api.html#storey.transformations..ForEach) | Applies the given function on each event in the stream, and passes the original event downstream. |
| [storey.transformations.MapClass](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.MapClass) | Similar to Map, but instead of a function argument, this class should be extended and its do() method overridden.  |
| [storey.transformations.MapWithState](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.MapWithState)  | Maps, or transforms, incoming events using a stateful user-provided function, and an initial state, which can be a database table.   |
| [storey.transformations.Partition](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.Partition) | Partitions events by calling a predicate function on each event. Each processed event results in a Partitioned namedtuple of (left=Optional[Event], right=Optional[Event]). |
| [storey.Reduce]() | Reduces incoming events into a single value that is returned upon the successful termination of the flow. |
[storey.transformations.SampleWindow](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.SampleWindow) | Emits a single event in a window of `window_size` events, in accordance with `emit_period` and `emit_before_termination`.   | 


## Data Transformations

| Class name            | Description                           |  
|----------------------------|--------------------------------------------------------------|   
| [storey.AggregateByKey](https://storey.readthedocs.io/en/latest/api.html#storey.aggregations.AggregateByKey) | Aggregates the data into the table object provided for later persistence, and outputs an event enriched with the requested aggregation features. |
| [DateExtractor]()  |              |
| [mlrun.feature_store.Imputer]() | Replace None values with default values.   |
| [mlrun.feature_store.MapValues]() | Map column values to new values.   |
| [mlrun.feature_store.OneHotEncoder]() | Create new binary fields, one per category (one hot encoded).  | 

## External IO and data enrichment
| Class name                                       | Description                                   |   
|--------------------------------------------------|---------------------------------|
| [BatchHttpRequests]() |                    | 
| [mlrun.datastore.DataItem](../api/mlrun.datastore.html#mlrun.datastore.DataItem) | Data input/output class abstracting access to various local/remote data sources |
| [storey.transformations.JoinWithTable](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.JoinWithTable) | Joins each event with data from the given table.  |
| [JoinWithV3IOTable]() | Joins each event with a V3IO table. Used for event augmentation.  | 
| [QueryByKey](https://storey.readthedocs.io/en/latest/api.html#module-storey.drivers.QueryByKey) | Similar to to AggregateByKey, but this step is for serving only and does not aggregate the event. | 
| [RemoteStep]() | Class for calling remote endpoints. | 
| [storey.transformations.SendToHttp](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.SendToHttp) | Joins each event with data from any HTTP source. Used for event augmentation. |
 

## Sources
| Class name                                       | Description                                   |   
|--------------------------------------------------|---------------------------------|
| [mlrun.datastore.BigQuerySource](../api/mlrun.datastore.html#mlrun.datastore.BigQuerySource) | Reads Google BigQuery query results as input source for a flow.  |
| [mlrun.datastore.CSVSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.CSVSource) | Reads a CSV file as input source for a flow.   |
| [DataframeSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.DataframeSource) | Reads data frame as input source for a flow. | 
| [mlrun.datastore.HttpSource](../api/mlrun.datastore.html#mlrun.datastore.HttpSource) |             |
| [mlrun.datastore.KafkaSource](../api/mlrun.datastore.html#mlrun.datastore.KafkaSource) | Sets the kafka source for the flow. |
| [mlrun.datastore.ParquetSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.ParquetSource) | Reads the Parquet file/dir as the input source for a flow.  |
| [mlrun.datastore.StreamSource](../api/mlrun.datastore.html#mlrun.datastore.StreamSource) | Sets the stream source for the flow. If the stream doesn’t exist it creates it.  | 


## Targets
| Class name                                       | Description                                   |   
|--------------------------------------------------|---------------------------------|
| [mlrun.datastore.CSVTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.CSVTarget) | Writes events to a CSV file. |
| [mlrun.datastore.NoSqlTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.NoSQLTarget)  |              |
| [mlrun.datastore.ParquetTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.ParquetTarget) | The Parquet target storage driver, used to materialize feature set/vector data into parquet files. |
| [mlrun.datastore.StreamTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.StreamTarget) | Writes all incoming events into a V3IO stream. |
| [storey.transformations.ToDataFrame](https://storey.readthedocs.io/en/latest/api.html#storey.transformations.ToDataFrame)  | Create pandas data frame from events. Can appear in the middle of the flow, as opposed to ReduceToDataFrame.| 
| [mlrun.datastore.TSBDTarget]() |  |

## Models
| Class name                                       | Description                                   |   
|--------------------------------------------------|---------------------------------|
| [mlrun.frameworks.onnx.ONNXModelServer]() | A model serving class for serving ONYX Models. A sub-class of the  V2ModelServer class. | 
| [mlrun.frameworks.pytorch.PyTorchModelServer]() | A model serving class for serving PyTorch Models. A sub-class of the  V2ModelServer class. |
| [mlrun.frameworks.sklearn.SklearnModelServer]() | A model serving class for serving Sklearn Models. A sub-class of the  V2ModelServer class. |  
| [mlrun.frameworks.tf_keras.TFKerasModelServer]() | A model serving class for serving TFKeras Models. A sub-class of the V2ModelServer class. |
| [mlrun.frameworks.xgboost.XGBModelServer]() | A model serving class for serving XGB Models. A sub-class of the  V2ModelServer class. | 

## Routers

| Class name                                       | Description                                                   |        
|--------------------------------------------------|---------------------------------------------------------------|
| [mlrun.serving.EnrichmentModelRouter]()  | Auto enrich the request with data from the feature store. The router input accepts a list of inference requests (each request can be a dict or a list of incoming features/keys). It enriches the request with data from the specified feature vector (`feature_vector_uri`). |
| [mlrun.serving.EnrichmentVotingEnsemble]()  | Auto enrich the request with data from the feature store. The router input accepts a list of inference requests (each request can be a dict or a list of incoming features/keys). It enriches the request with data from the specified feature vector (`feature_vector_uri`). |
| [mlrun.serving.ModelRouter]() | Basic model router, for calling different models per each model path.  | 
| [mlrun.serving.VotingEnsemble](../api/mlrun.serving.html#mlrun.serving.VotingEnsemble)  | An ensemble machine learning model that combines the prediction of several models. |       


## Other
| Class name                                       | Description                                   |   
|--------------------------------------------------|---------------------------------|
| [mlrun.feature_store.FeaturesetValidator]() | Validate feature values according to the feature set validation policy. | 
| [ReduceToDataFrame]() |          |