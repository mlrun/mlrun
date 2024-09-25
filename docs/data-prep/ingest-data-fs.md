(ingest-data-fs)=
# Ingest data using the feature store

Define the source and material targets, and start the ingestion process (as [local process](#ingest-data-locally), [using an MLRun job](#ingest-data-using-an-mlrun-job), [real-time ingestion](#real-time-ingestion), or [incremental ingestion](#incremental-ingestion)).

Data can be ingested as a batch process either by running the ingest command on demand or as a scheduled job. Batch ingestion 
can be done locally (i.e. running as a python process in the Jupyter pod) or as an MLRun job.

The data source can be a DataFrame or files (e.g. csv, parquet). Files can be either local files residing on a volume (e.g. v3io), 
or remote (e.g. S3, Azure blob). MLRun also supports Google BigQuery as a data source. If you define a transformation graph, then 
the ingestion process runs the graph transformations, infers metadata and stats, and writes the results to a target data store.

When targets are not specified, data is stored in the configured default targets (i.e. NoSQL for real-time and Parquet for offline).

## Ingestion engines

MLRun supports a several ingestion engines:
- `storey` engine (default) is designed for real-time data (e.g. individual records) that will be transformed using Python functions and classes
- `pandas` engine is designed for batch data that can fit into memory that will be transformed using Pandas dataframes. Pandas is used for testing, and is not recommended for production deployments
- `spark` engine is designed for batch data.


```{admonition} Limitations
- Do not name columns starting with either `_` or `aggr_`. They are reserved for internal use. See 
also general limitations in [Attribute name restrictions](https://www.iguazio.com/docs/latest-release/data-layer/objects/attributes/#attribute-names).
- Do not name columns to match the regex pattern `.*_[a-z]+_[0-9]+[smhd]$`, where [a-z]+ is an aggregation name,
one of: count, sum, sqr, max, min, first, last, avg, stdvar, stddev. E.g. x_count_1h.
- When using the pandas engine, do not use spaces (` `) or periods (`.`) in the column names. These cause errors in the ingestion.
```

**In this section**
- [Verify a feature set with a small dataset by inferring data](#verify-a-feature-set-with-a-small-dataset-by-inferring-data)
- [Ingest data locally](#ingest-data-locally)
- [Ingest data using an MLRun job](#ingest-data-using-an-mlrun-job)
- [Real-time ingestion](#real-time-ingestion)
- [Incremental ingestion](#incremental-ingestion)

**See also**:
- {ref}`feature-sets`
- {ref}`sources-targets`

## Verify a feature set with a small dataset by inferring data 

Ingesting an entire dataset can take a fair amount of time. Before ingesting the entire dataset,  you can check the feature 
set definition by 
simulating the creation of the feature set. <br>
This gives a preview of the results (in the returned dataframe). The simulation method is called `infer`. 
It infers the source data schema, and processes the graph logic (assuming there is one) on a small subset of data. 
The infer operation also learns the feature set schema and, by default, does statistical analysis on the result.
  
```python
df = quotes_set.preview(quotes)

# print the feature statistics
print(quotes_set.get_stats_table())
```

### Inferring data

There are two ways to infer data:
- Metadata/schema: This is responsible for describing the dataset and generating its meta-data, such as deducing the 
data-types of the features and listing the entities that are involved. Options belonging to this type are 
`Entities`, `Features` and `Index`. The `InferOptions` class has the `InferOptions.schema()` function that returns a value 
containing all the options of this type.
- Stats/preview: This relates to calculating statistics and generating a preview of the actual data in the dataset. 
Options of this type are `Stats`, `Histogram` and `Preview`. 

The `InferOptions class` has the following values:<br>
class InferOptions:<br>
    Null = 0<br>
    Entities = 1<br>
    Features = 2<br>
    Index = 4<br>
    Stats = 8<br>
    Histogram = 16<br>
    Preview = 32<br>
    
The `InferOptions class` basically translates to a value that can be a combination of the above values. For example, passing a value of 
24 means `Stats` + `Histogram`.

When simultaneously ingesting data and requesting infer options, part of the data might be ingested twice: once for inferring 
metadata/stats and once for the actual ingest. This is normal behavior.

## Ingest data locally

Use a feature set to create the basic feature-set definition and then an ingest method to run a simple ingestion "locally" in the Jupyter Notebook pod.

```python
# Simple feature set that reads a csv file as a dataframe and ingests it "as is"
stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
stocks = pd.read_csv("stocks.csv")
df = stocks_set.ingest(stocks)

# Specify a csv file as source, specify a custom CSV target
source = CSVSource("mycsv", path="stocks.csv")
targets = [CSVTarget("mycsv", path="./new_stocks.csv")]
measurements.ingest(source, targets)
```
You can **update a feature set** either by overwriting its data (`overwrite=true`), or by appending data (`overwrite=false`). 
To append data you need to reuse the feature set that was used in previous ingestions 
that was saved in the DB (and not create a new feature set on every ingest).<br>
For example:
```python
try:
    my_fset = fstore.get_feature_set("my_fset")
except mlrun.errors.MLRunNotFoundError:
    my_fset = FeatureSet("my_fset", entities=[Entity("key")])

my_fset.ingest(overwrite=false)
```

To learn more about ingest, go to {py:class}`~mlrun.feature_store.ingest`.

## Ingest data using an MLRun job

Use the ingest method with the `run_config` parameter for running the ingestion process using a serverless MLRun job. <br>
By doing that, the ingestion process runs on its own pod or service on the kubernetes cluster. <br>
This option is more robust since it can leverage the cluster resources, as opposed to running within the Jupyter Notebook.<br>
It also enables you to schedule the job or use bigger/faster resources.

```python
# Running as a remote job
stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
config = RunConfig(image="mlrun/mlrun")
df = stocks_set.ingest(stocks, run_config=config)
```

## Real-time ingestion

Real-time use cases (e.g. real-time fraud detection) require feature engineering on live data (e.g. z-score calculation)
while the data is coming from a streaming engine (e.g. Kafka) or a live http endpoint. <br>
The feature store enables you to start real-time ingestion service. <br>
When running the {py:class}`~mlrun.feature_store.deploy_ingestion_service` the feature store creates an elastic real-time serverless function 
(the Nuclio function) that runs the pipeline and stores the data results in the "offline" and "online" feature store by default. <br>
There are multiple data source options including HTTP, Kafka, Kinesis, v3io stream, etc. <br>
Due to the asynchronous nature of feature store's execution engine, errors are not returned, but rather logged and pushed to the defined
error stream. <br>
```python
# Create a project, then a real time function that receives http requests
# the "ingest" function runs the feature engineering logic on live events
project = mlrun.get_or_create_project("real-time")
source = HTTPSource()
func = project.set_function(name="ingest", kind="serving").apply(mount_v3io())
config = RunConfig(function=func)
my_set.deploy_ingestion_service(source, run_config=config)
```

To learn more about `deploy_ingestion_service` go to {py:class}`~mlrun.feature_store.deploy_ingestion_service`.

## Incremental ingestion

You can schedule an ingestion job for a feature set on an ongoing basis. The first scheduled job runs on all the data in the source 
and the subsequent jobs ingest only the deltas since the previous run (from the last timestamp of the previous run until `datetime.now`). 
Example:

```
cron_trigger = "* */1 * * *" # will run every hour
fs = fstore.FeatureSet("stocks", entities=[fstore.Entity("ticker")])
fs.ingest(
    source=ParquetSource("mypq", path="stocks.parquet", time_field="time", schedule=cron_trigger),
    run_config=fstore.RunConfig(image='mlrun/mlrun')
)
```

The default value for the `overwrite` parameter in the ingest function for scheduled ingest is `False`, meaning that the 
target from the previous ingest is not deleted.
For the storey and pandas ingestion engines, the feature is currently implemented for ParquetSource only (CsvSource will be supported 
in a future release). For Spark engine both ParquetSource and CsvSource are supported.


