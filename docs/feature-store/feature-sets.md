# Feature sets

In MLRun, a group of features can be ingested together and stored in logical group called feature set. 
feature set take data from offline or online sources, build a list of features through a set of transformations, and 
store the resulting features along with the associated metadata and statistics. <br>
A feature set can be viewed as a database table with multiple material implementations for batch and real-time access,
along with the data pipeline definitions used to produce the features.
 
The feature set object contains the following information:
- **Metadata** - General information which is helpful for search and organization. Examples are project, name, owner, last update, description, labels and etc..
- **Key attributes** - Entity (the join key), timestamp key (optional), label column.
- **Features** - the list of features along with their schema, metadata, validation policies and statistics
- **Source** - The online or offline data source definitions and ingestion policy (file, database, stream, http endpoint, ..).
- **Transformation** - The data transformation pipeline (e.g. aggregation, enrichment etc..).
- **Target stores** - The type (i.e. parquet/csv or key value), location and status for the feature set materialized data. 
- **Function** - the type (storey, pandas, spark) and attributes of the data pipeline serverless functions.

## Building and Using Feature Sets

Creating a feature set comprises of the following steps:
* Create a new {py:class}`~mlrun.feature_store.FeatureSet` with the base definitions (name, entities, engine, etc.)
* Define the data processing steps using a transformations graph (DAG)
* Simulate and debug the data pipeline with a small dataset
* define the source and material targets, and start the ingestion process (as local process, remote job, 
  or real-time function)

### Create a FeatureSet:
* **name** &mdash; The feature set name is a unique name within a project. 
* **entities** &mdash; Each feature set must be associated with one or more index column. when joining feature sets the entity is used as the key column.
* **timestamp_key** &mdash; (optional) - it is used for specifiying the time field when joining by time
* **engine** &mdash; the processing engine type (storey, pandas, spark)

Example:
```python
#Create a basic feature set example
stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
```

To learn more about FeatureSet go to {py:class}`~mlrun.feature_store.FeatureSet` 

### Add Transformations 

Feature set data pipeline take raw data from online or offline sources and transforms it to meaningful features,
MLRun feature store support three processing engines (storey, pandas, spark) which can run in the client 
(e.g. Notebook) for interactive development or in elastic serverless functions for production and scale.

The data pipeline is defined using MLRun graph (DAG) language, graph steps can be pre-defined operators 
(such as aggregate, filter, encode, map, join, impute, etc) or custom python classes/functions. 
Read more about the graph in [**The Graph State Machine**](../serving/serving-graph.md#the-graph-state-machine)

the `pandas` and `spark` engines are good for simple batch transformations while the `storey` stream processing engine (the default engine)
can handle complex workflows and real-time sources.

The results from the transformation pipeline are stored in one or more material targets, usually data for offline 
access such as training will be stored in Parquet files and data for online access such as serving will be stored 
in a NoSQL DB, users can use the default targets or add/replace with additional custom targets.

Graph example (storey engine):
```python
feature_set = FeatureSet("measurements", entities=[Entity(key)], timestamp_key="timestamp")
# Define the computational graph including our custom functions
feature_set.graph.to(DropColumns(drop_columns))\
                 .to(RenameColumns(mapping={'bad': 'bed'}))
feature_set.add_aggregation('hr', 'hr', ['avg'], ["1h"])
feature_set.plot()
fs.ingest(feature_set, data_df)
```

Graph example (pandas engine):
```python
def myfunc1(df, context=None):
    df = df.drop(columns=["exchange"])
    return df

stocks_set = fs.FeatureSet("stocks", entities=[Entity("ticker")], engine="pandas")
stocks_set.graph.to(name="s1", handler="myfunc1")
df = fs.ingest(stocks_set, stocks_df)
```

The graph steps can use built-in transformation classes, simple python classes or function handlers. 

### Simulate The Data Pipeline
During the development phase it's pretty common to check the feature set definition and simulate the creation of the feature set before ingesting the entire dataset which can take time. <br>
This allows to get a preview of the results (in the returned dataframe). The simulation method is called `infer`, it infers the source data schema as well as processing the graph logic (assuming there is one) on a small subset of data. 
The infer operation also learns the feature set schema and does statistical analysis on the result by default.
  
```python
df = fs.infer(quotes_set, quotes)

# print the featue statistics
print(quotes_set.get_stats_table())
```

## Ingest Data Into The Feature Store

Data can be ingested as a batch process either by running the ingest command on demand or as a scheduled job.
The data source could be a DataFrame or files (e.g. csv, parquet). Files can be either local files residing on a volume (e.g. v3io) or remote (e.g. S3, Azure blob). If the user defines a transfomration graph then when running an ingestion process it runs the graph transformations, infers metadata and stats and writes the results to a target data store.
When targets are not specified data is stored in the configured default targets (i.e. NoSQL for real-time and Parquet for offline).
Batch ingestion can be done locally (i.e. running as a python process in the Jupyter pod) or as an MLRun job.

#### Ingest data (locally)

Use FeatureSet to create the basic feature set definition and then the ingest method to run a simple ingestion "localy" in the jupyter notebook pod.


```python
# Simple feature set that reads a csv file as a dataframe and ingest it as is 
stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
stocks = pd.read_csv("stocks.csv")
df = ingest(stocks_set, stocks)

# specify a csv file as source, specify custom CSV target 
source = CSVSource("mycsv", path="stocks.csv")
targets = [CSVTarget("mycsv", path="./new_stocks.csv")]
ingest(measurements, source, targets)
```

To learn more about ingest go to {py:class}`~mlrun.feature_store.ingest`

#### Ingest data using an MLRun job

Use the ingest method with run_config parameter for running the ingestion process using a serverless MLrun job. <br>
By doing that, the ingestion process is running on its own pod or service on the kubernetes cluster. <br>
Using this option is more robust as it can leverage the cluster resources as opposed to running within the jupyter notebook.<br>
It also enables users to schedule the job or use bigger/faster resources.

```python
# running as remote job
stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
config = RunConfig(image='mlrun/mlrun').apply(mount_v3io())
df = ingest(stocks_set, stocks, run_config=config)
```

### Real time ingestion

Real time use cases (e.g. real time fraud detection) requires feature engineering on live data (e.g. z-score calculation)
while the data is coming from a streaming engine (e.g. kafka) or a live http endpoint. <br>
The feature store enables users to start real-time ingestion service. <br>
When running the {py:class}`~mlrun.feature_store.deploy_ingestion_service` the feature store creates an elastic real time serverless function 
(AKA nuclio function) which runs the pipeline and stores the data results in the "offline" and "online" feature store by default. <br>
There are multiple data source options including http, kafka, kinesis, v3io stream, etc. <br>

```python
# Create a real time function that recieve http requests
# the "ingest" function runs the feature engineering logic on live events
source = HTTPSource()
func = mlrun.code_to_function("ingest", kind="serving").apply(mount_v3io())
config = RunConfig(function=func)
fs.deploy_ingestion_service(my_set, source, run_config=config)
```

To learn more about deploy_ingestion_service go to {py:class}`~mlrun.feature_store.deploy_ingestion_service` 

### Data sources

For batch ingestion the feature store supports dataframes or files (i.e. csv & parquet). <br>
The files can reside on S3, NFS, Azure blob storage or on Iguazio platform. <br>
For real time ingestion the source could be http, kafka or v3io stream, etc.
When defining a source  it maps to a nuclio event triggers. <br>

Note that users can also create a custom `source` to access various databases or data sources.

### Target stores
By default the feature sets are stored as both parquet file for training and as a key value table (in Iguazio platform) for online serving. <br>
The parquet file is ideal for fetching large set of data for training while the key value is ideal for an online application as it supports low latency data retrieval based on key access. <br>

> **Note:** When working with Iguazio platform the default feature set storage location is under "Projects" container --> <project name>/fs/.. folder. 
the default location can be modified in mlrun config or specified per injest operation. the parquet/csv files can be stored in NFS, S3, Azure blob storage and on Iguazio DB/FS.

