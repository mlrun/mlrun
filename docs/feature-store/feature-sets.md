# Feature sets

In MLRun, a group of features can be ingested together and stored in logical group called feature set. 
Feature sets take data from offline or online sources, build a list of features through a set of transformations, and 
store the resulting features along with the associated metadata and statistics. <br>
A feature set can be viewed as a database table with multiple material implementations for batch and real-time access,
along with the data pipeline definitions used to produce the features.
 
The feature set object contains the following information:
- **Metadata**&mdash;General information which is helpful for search and organization. Examples are project, name, owner, last update, description, labels and etc..
- **Key attributes**&mdash;Entity (the join key), timestamp key (optional), label column.
- **Features**&mdash;the list of features along with their schema, metadata, validation policies and statistics
- **Source**&mdash;The online or offline data source definitions and ingestion policy (file, database, stream, http endpoint, etc.).
- **Transformation**&mdash;The data transformation pipeline (e.g. aggregation, enrichment etc.).
- **Target stores**&mdash;The type (i.e. parquet/csv or key value), location and status for the feature set materialized data. 
- **Function**&mdash;The type (storey, pandas, spark) and attributes of the data pipeline serverless functions.

## Building and Using Feature Sets

Creating a feature set comprises of the following steps:
* Create a new {py:class}`~mlrun.feature_store.FeatureSet` with the base definitions (name, entities, engine, etc.).
* Define the data processing steps using a transformations graph (DAG).
* Simulate and debug the data pipeline with a small dataset.
* Define the source and material targets, and start the ingestion process (as local process, remote job, 
  or real-time function).

### Create a FeatureSet
* **name**&mdash;The feature set name is a unique name within a project. 
* **entities**&mdash;Each feature set must be associated with one or more index column. When joining feature sets, the entity is used as the key column.
* **timestamp_key**&mdash;(optional) Used for specifying the time field when joining by time.
* **engine**&mdash;The processing engine type:
   - Spark
   - pandas
   - storey (some advanced functionalities are in the Beta state)
   
Example:
```python
#Create a basic feature set example
stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
```

To learn more about Feature Sets go to {py:class}`~mlrun.feature_store.FeatureSet`.

```{admonition} Note 
Feature sets can also be created in the UI. To create a feature set:
1. Select a project and press **Feature store**, then press **Create Set**.
3. After completing the form, press **Save and Ingest** to start the process, or **Save** to save the set for later ingestion.
```

### Add Transformations 

A feature set data pipeline takes raw data from online or offline sources and transforms it to meaningful features.
The MLRun feature store supports three processing engines (storey, pandas, spark) that can run in the client 
(e.g. Notebook) for interactive development or in elastic serverless functions for production and scale.

The data pipeline is defined using MLRun graph (DAG) language. Graph steps can be pre-defined operators 
(such as aggregate, filter, encode, map, join, impute, etc) or custom python classes/functions. 
Read more about the graph in [**Serving and Data Pipelines**](../serving/serving-graph.md).

The `pandas` and `spark` engines are good for simple batch transformations, while the `storey` stream processing engine (the default engine)
can handle complex workflows and real-time sources.

The results from the transformation pipeline are stored in one or more material targets.  Data for offline 
access, such as training, is usually stored in Parquet files. Data for online access such as serving is stored 
in a NoSQL DB. Users can use the default targets or add/replace with additional custom targets.

Graph example (storey engine):
```python
import mlrun.feature_store as fstore
feature_set = fstore.FeatureSet("measurements", entities=[Entity(key)], timestamp_key="timestamp")
# Define the computational graph including the custom functions
feature_set.graph.to(DropColumns(drop_columns))\
                 .to(RenameColumns(mapping={'bad': 'bed'}))
feature_set.add_aggregation('hr', ['avg'], ["1h"])
feature_set.plot()
fstore.ingest(feature_set, data_df)
```

Graph example (pandas engine):
```python
def myfunc1(df, context=None):
    df = df.drop(columns=["exchange"])
    return df

stocks_set = fstore.FeatureSet("stocks", entities=[Entity("ticker")], engine="pandas")
stocks_set.graph.to(name="s1", handler="myfunc1")
df = fstore.ingest(stocks_set, stocks_df)
```

The graph steps can use built-in transformation classes, simple python classes, or function handlers. 

### Simulate The Data Pipeline
During the development phase it's pretty common to check the feature set definition and to simulate the creation of the feature set before ingesting the entire dataset, since ingesting the entire feature set can take time. <br>
This allows you to get a preview of the results (in the returned dataframe). The simulation method is called `infer`. It infers the source data schema as well as processing the graph logic (assuming there is one) on a small subset of data. 
The infer operation also learns the feature set schema and does statistical analysis on the result by default.
  
```python
df = fstore.preview(quotes_set, quotes)

# print the featue statistics
print(quotes_set.get_stats_table())
```

## Ingest Data into the Feature Store

Data can be ingested as a batch process either by running the ingest command on demand or as a scheduled job.
The data source could be a DataFrame or files (e.g. csv, parquet). Files can be either local files residing on a volume (e.g. v3io), and remote (e.g. S3, Azure blob). MLRun also supports Google BigQuery as a data source. If you define a transformation graph, then the ingestion process runs the graph transformations, infers metadata and stats, and writes the results to a target data store.
When targets are not specified, data is stored in the configured default targets (i.e. NoSQL for real-time and Parquet for offline).
Batch ingestion can be done locally (i.e. running as a python process in the Jupyter pod) or as an MLRun job.

```{admonition} Limitation
- Do not name columns starting with either `t_` or `aggr_`. They are reserved for internal use, and the data does not ingest correctly. See also general limitations in [Attribute name restrictions](https://www.iguazio.com/docs/latest-release/data-layer/objects/attributes/#attribute-names).
- When using the pandas engine, do not use spaces (` `) or periods (`.`) in the column names. These cause errors in the ingestion.
```

### Ingest data (locally)

Use a Feature Set to create the basic feature-set definition and then an ingest method to run a simple ingestion "locally" in the Jupyter Notebook pod.

```python
# Simple feature set that reads a csv file as a dataframe and ingests it as is 
stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
stocks = pd.read_csv("stocks.csv")
df = ingest(stocks_set, stocks)

# Specify a csv file as source, specify a custom CSV target 
source = CSVSource("mycsv", path="stocks.csv")
targets = [CSVTarget("mycsv", path="./new_stocks.csv")]
ingest(measurements, source, targets)
```

To learn more about ingest go to {py:class}`~mlrun.feature_store.ingest`.

### Ingest data using an MLRun job

Use the ingest method with the `run_config` parameter for running the ingestion process using a serverless MLRun job. <br>
By doing that, the ingestion process runs on its own pod or service on the kubernetes cluster. <br>
This option is more robust since it can leverage the cluster resources, as opposed to running within the Jupyter Notebook.<br>
It also enables you to schedule the job or use bigger/faster resources.

```python
# Running as remote job
stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
config = RunConfig(image='mlrun/mlrun')
df = ingest(stocks_set, stocks, run_config=config)
```

### Real time ingestion

Real time use cases (e.g. real time fraud detection) require feature engineering on live data (e.g. z-score calculation)
while the data is coming from a streaming engine (e.g. kafka) or a live http endpoint. <br>
The feature store enables you to start real-time ingestion service. <br>
When running the {py:class}`~mlrun.feature_store.deploy_ingestion_service` the feature store creates an elastic real time serverless function 
(the nuclio function) that runs the pipeline and stores the data results in the "offline" and "online" feature store by default. <br>
There are multiple data source options including http, kafka, kinesis, v3io stream, etc. <br>
Due to the asynchronous nature of feature store's execution engine, errors are not returned, but rather logged and pushed to the defined
error stream. <br>
```python
# Create a real time function that receives http requests
# the "ingest" function runs the feature engineering logic on live events
source = HTTPSource()
func = mlrun.code_to_function("ingest", kind="serving").apply(mount_v3io())
config = RunConfig(function=func)
fstore.deploy_ingestion_service(my_set, source, run_config=config)
```

To learn more about deploy_ingestion_service go to {py:class}`~mlrun.feature_store.deploy_ingestion_service`.

### Incremental ingestion

You can schedule an ingestion job for a feature set on an ongoing basis. The first scheduled job runs on all the data in the source and the subsequent jobs ingest only the deltas since the previous run (from the last timestamp of the previous run until `datetime.now`). 
Example:

`cron_trigger = "* */1 * * *" #will run every hour
source = ParquetSource("myparquet", path=path, time_field="time", schedule=cron_trigger)
feature_set = fs.FeatureSet(name=name, entities=[fs.Entity("first_name")], timestamp_key="time",)
fs.ingest(feature_set, source, run_config=fs.RunConfig())`

The default value for the `overwrite` parameter in the ingest function for scheduled ingest is `False`, meaning that the target from the previous ingest is not deleted.
For the storey engine, the feature is currently implemented for ParquetSource only. (CsvSource will be supported in a future release). For Spark engine, other sources are also supported. 

### Data sources

For batch ingestion the feature store supports dataframes or files (i.e. csv & parquet). <br>
The files can reside on S3, NFS, Azure blob storage, or the Iguazio platform. MLRun also supports Google BigQuery as a data source. 
When working with S3/Azure, there are additional requirements. Use pip install mlrun[s3] or pip install mlrun[azure-blob-storage] to install them. 
- Azure: define the environment variable `AZURE_STORAGE_CONNECTION_STRING`. 
- S3: define `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_BUCKET`.

For real time ingestion the source can be http, kafka or v3io stream, etc.
When defining a source, it maps to nuclio event triggers. <br>

Note that you can also create a custom `source` to access various databases or data sources.

### Target stores
By default the feature sets are stored as both parquet file for training and as a key value table (in the Iguazio MLOps platform) for online serving. <br>
The parquet file is ideal for fetching large set of data for training while the key value is ideal for an online application since it supports low latency data retrieval based on key access. <br>

```{admonition} Note
When working with the Iguazio MLOps platform the default feature set storage location is under the "Projects" container: <project name>/fs/.. folder. 
The default location can be modified in mlrun config or specified per ingest operation. The parquet/csv files can be stored in NFS, S3, Azure blob storage and on Iguazio DB/FS.
```
