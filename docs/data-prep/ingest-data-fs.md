(ingest-data-fs)=
# Ingest data using the feature store

Define the source and material targets, and start the ingestion process (as [local process](#ingest-data-locally), [using an MLRun job](#ingest-data-using-an-mlrun-job), [real-time ingestion](#real-time-ingestion), or [incremental ingestion](#incremental-ingestion)).

Data can be ingested as a batch process either by running the ingest command on demand or as a scheduled job. Batch ingestion 
can be done locally (i.e. running as a python process in the Jupyter pod) or as an MLRun job.

The data source can be a DataFrame or files (e.g. csv, parquet). Files can be either local files residing on a volume (e.g. v3io), 
or remote (e.g. S3, Azure blob). MLRun also supports Google BigQuery as a data source. If you define a transformation graph, then 
the ingestion process runs the graph transformations, infers metadata and stats, and writes the results to a target data store.

When targets are not specified, data is stored in the configured default targets (i.e. NoSQL for real-time and Parquet for offline).


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
- [Data sources](#data-sources)
- [Target stores](#target-stores)

**See also**:
- {ref}`feature-sets`

## Verify a feature set with a small dataset by inferring data 

Ingesting an entire dataset can take a fair amount of time. Before ingesting the entire dataset,  you can check the feature 
set definition by 
simulating the creation of the feature set. <br>
This gives a preview of the results (in the returned dataframe). The simulation method is called `infer`. 
It infers the source data schema, and processes the graph logic (assuming there is one) on a small subset of data. 
The infer operation also learns the feature set schema and, by default, does statistical analysis on the result.
  
```python
df = fstore.preview(quotes_set, quotes)

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
df = ingest(stocks_set, stocks)

# Specify a csv file as source, specify a custom CSV target 
source = CSVSource("mycsv", path="stocks.csv")
targets = [CSVTarget("mycsv", path="./new_stocks.csv")]
ingest(measurements, source, targets)
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
config = RunConfig(image='mlrun/mlrun')
df = ingest(stocks_set, stocks, run_config=config)
```

## Real-time ingestion

Real-time use cases (e.g. real-time fraud detection) require feature engineering on live data (e.g. z-score calculation)
while the data is coming from a streaming engine (e.g. kafka) or a live http endpoint. <br>
The feature store enables you to start real-time ingestion service. <br>
When running the {py:class}`~mlrun.feature_store.deploy_ingestion_service` the feature store creates an elastic real-time serverless function 
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

To learn more about `deploy_ingestion_service` go to {py:class}`~mlrun.feature_store.deploy_ingestion_service`.

## Incremental ingestion

You can schedule an ingestion job for a feature set on an ongoing basis. The first scheduled job runs on all the data in the source 
and the subsequent jobs ingest only the deltas since the previous run (from the last timestamp of the previous run until `datetime.now`). 
Example:

```
cron_trigger = "* */1 * * *" #runs every hour
source = ParquetSource("myparquet", path=path, schedule=cron_trigger)
feature_set = fstore.FeatureSet(name=name, entities=[fstore.Entity("first_name")], timestamp_key="time",)
fstore.ingest(feature_set, source, run_config=fstore.RunConfig())
```

The default value for the `overwrite` parameter in the ingest function for scheduled ingest is `False`, meaning that the 
target from the previous ingest is not deleted.
For the storey engine, the feature is currently implemented for ParquetSource only. (CsvSource will be supported in a future release). 
For Spark engine, other sources are also supported. 

## Data sources

For batch ingestion the feature store supports dataframes and files (i.e. csv & parquet). <br>
The files can reside on S3, NFS, SQL (for example, MYSQL), Azure blob storage, or the Iguazio platform. MLRun also supports Google BigQuery as a data source. 

For real time ingestion the source can be http, Kafka, MySQL, or V3IO stream, etc.
When defining a source, it maps to nuclio event triggers. <br>

You can also create a custom `source` to access various databases or data sources.

### S3/Azure data source

When working with S3/Azure, there are additional requirements. Use: pip install mlrun[s3]; pip install mlrun[azure-blob-storage]; 
or pip install mlrun[google-cloud-storage] to install them. 
- Azure: define the environment variable `AZURE_STORAGE_CONNECTION_STRING`
- S3: define `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_BUCKET`

### SQL data source

```{admonition} Note
Tech Preview 
```
```{admonition} Limitation
Do not use SQL reserved words as entity names. See more details in [Keywords and Reserved Words](https://dev.mysql.com/doc/refman/8.0/en/keywords.html).
```
`SQLSource` can be used for both batch ingestion and real time ingestion. It supports storey but does not support Spark. To configure 
either, pass the `db_uri` or overwrite the `MLRUN_SQL__URL` env var, in this format:<br> 
`mysql+pymysql://<username>:<password>@<host>:<port>/<db_name>`, for example:

```
source = SQLSource(table_name='my_table', 
                     db_path="mysql+pymysql://abc:abc@localhost:3306/my_db", 
                     key_field='key',
                     time_fields=['timestamp'], )
 
 feature_set = fs.FeatureSet("my_fs", entities=[fs.Entity('key')],)
 feature_set.set_targets([])
 df = fs.ingest(feature_set, source=source)
```

### Apache Kafka data source

Example:

```
from mlrun.datastore.sources import KafkaSource


with open('/v3io/bigdata/name.crt') as x: 
    caCert = x.read()  
caCert

kafka_source = KafkaSource(
            brokers=['default-tenant.app.vmdev76.lab.iguazeng.com:9092'],
            topics="stocks-topic",
            initial_offset="earliest",
            group="my_group",
        )
        
run_config = fstore.RunConfig(local=False).apply(mlrun.auto_mount())

stocks_set_endpoint = fstore.deploy_ingestion_service(featureset=stocks_set, source=kafka_source,run_config=run_config)
```        
       

### Confluent Kafka data source

```{admonition} Note
Tech Preview 
```

Example:

```
from mlrun.datastore.sources import KafkaSource


with open('/v3io/bigdata/name.crt') as x: 
    caCert = x.read()  
caCert


kafka_source = KafkaSource(
        brokers=['server-1:9092', 
        'server-2:9092', 
        'server-3:9092', 
        'server-4:9092', 
        'server-5:9092'],
        topics=["topic-name"],
        initial_offset="earliest",
        group="test",
        attributes={"sasl" : {
                      "enable": True,
                      "password" : "pword",
                      "user" : "user",
                      "handshake" : True,
                      "mechanism" : "SCRAM-SHA-256"},
                    "tls" : {
                      "enable": True,
                      "insecureSkipVerify" : False
                    },            
                   "caCert" : caCert}
    )
    
run_config = fstore.RunConfig(local=False).apply(mlrun.auto_mount())

stocks_set_endpoint = fstore.deploy_ingestion_service(featureset=stocks_set, source=kafka_source,run_config=run_config)
```


## Target stores

By default, the feature sets are saved in parquet and the Iguazio NoSQL DB ({py:class}`~mlrun.datastore.NoSqlTarget`). <br>
The Parquet file is ideal for fetching large set of data for training while the key value is ideal for an online application 
since it supports low latency data retrieval based on key access. 

```{admonition} Note
When working with the Iguazio MLOps platform the default feature set storage location is under the "Projects" container: `<project name>/fs/..` folder. 
The default location can be modified in mlrun config or specified per ingest operation. The parquet/csv files can be stored in 
NFS, S3, Azure blob storage, Redis, SQL, and on Iguazio DB/FS.
```

### Redis target store

```{admonition} Note
Tech Preview
```

The Redis online target is called, in MLRun, `RedisNoSqlTarget`. The functionality of the `RedisNoSqlTarget` is identical to the `NoSqlTarget` except for:
- The RedisNoSqlTarget accepts the path parameter in the form: `<redis|rediss>://<host>[:port]`
For example: `rediss://localhost:6379` creates a redis target, where:
   - The client/server protocol (rediss) is TLS protected (vs. "redis" if no TLS is established)
   - The server location is localhost port 6379.
- If the path parameter is not set, it tries to fetch it from the MLRUN_REDIS__URL environment variable.
- You cannot pass the username/password as part of the URL. If you want to provide the username/password, use secrets as:
`<prefix_>REDIS_USER <prefix_>REDIS_PASSWORD` where \<prefix> is the optional RedisNoSqlTarget `credentials_prefix` parameter.
- Two types of Redis servers are supported: StandAlone and Cluster (no need to specify the server type in the config).
- A feature set supports one online target only. Therefore `RedisNoSqlTarget` and `NoSqlTarget` cannot be used as two targets of the same feature set.
    
The K8s secrets are not available when executing locally (from the sdk). Therefore, if RedisNoSqlTarget with secret is used, 
You must add the secret as an env-var.

To use the Redis online target store, you can either change the default to be parquet and Redis, or you can specify the Redis target 
explicitly each time with the path parameter, for example:</br>
`RedisNoSqlTarget(path ="redis://1.2.3.4:6379")`

### SQL target store

```{admonition} Note
Tech Preview 
```
```{admonition} Limitation
Do not use SQL reserved words as entity names. See more details in [Keywords and Reserved Words](https://dev.mysql.com/doc/refman/8.0/en/keywords.html).
```
The `SQLTarget` online target supports storey but does not support Spark. Aggregations are not supported.<br>
To configure, pass the `db_uri` or overwrite the `MLRUN_SQL__URL` env var, in this format:<br>
`mysql+pymysql://<username>:<password>@<host>:<port>/<db_name>`

You can pass the schema and the name of the table you want to create or the name of an existing table, for example:

```
 target = SQLTarget(
            table_name='my_table',
            schema= {'id': string, 'age': int, 'time': pd.Timestamp, ...}
            create_table=True,
            primary_key_column='id',
            time_fields=["time"]
        )
feature_set = fs.FeatureSet("my_fs", entities=[fs.Entity('id')],)
fs.ingest(feature_set, source=df, targets=[target])
```