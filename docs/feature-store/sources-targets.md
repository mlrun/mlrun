(sources-targets)=
# Sources and targets

- [Sources](#sources)
- [Targets](#targets)

# Sources

For batch ingestion the feature store supports dataframes and files (i.e. csv & parquet). <br>
The files can reside on S3, NFS, SQL (for example, MYSQL), Azure blob storage, or the Iguazio platform. MLRun also supports Google BigQuery as a data source. 

For real time ingestion the source can be http, Kafka, MySQL, or V3IO stream, etc.
When defining a source, it maps to Nuclio event triggers. <br>

You can also create a custom `source` to access various databases or data sources.

| Class name                                                                                         | Description                                                   | storey | spark | pandas |
|----------------------------------------------------------------------------------------------------| ---------------------------------                              | ---    | ---   | ---    |
| {py:meth}`~mlrun.datastore.BigQuerySource`                                                         | Batch. Reads Google BigQuery query results as input source for a flow.| N      | Y     | Y      |
| [SnowFlakeSource](#snowflake-source)                                                               | Batch. Reads Snowflake query results as input source for a flow         | N      | Y     | N      |
| [SQLSource](#sql-source)                                                                           | Batch. Reads SQL query results as input source for a flow               | Y      | N     | Y      |
| {py:meth}`~mlrun.datastore.CSVSource`                                                              | Batch. Reads a CSV file as input source for a flow.                   | Y      | Y     | Y      |
| [DataframeSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.DataframeSource) | Batch. Reads data frame as input source for a flow.                   | Y      | N     | N      |
| [ParquetSource](#parquet-source)                                                                   | Batch. Reads the Parquet file/dir as the input source for a flow.     | Y      | Y     | Y      |
| [S3/Azure source](#s3-azure-source)                                                                | Batch.                                                                 |       |      |       |
| {py:meth}`~mlrun.datastore.HttpSource`                                                             |Event-based. Sets the HTTP-endpoint source for the flow.    | Y      | N     | N      |
| [Kafka source](#kafka-source)                                                                      |Event-based. Sets a Kafka source for the flow (supports both Apache and Confluence Kafka).| Y      | N     | N      |
| {py:meth}`~mlrun.datastore.StreamSource`                                                           |Event-based. Sets the stream source for the flow. If the stream doesn’t exist it creates it. | Y      | N     | N      |

## Snowflake source
An example of SnowflakeSource ingest:
```python
os.environ["SNOWFLAKE_PASSWORD"] = "*****"
source = SnowflakeSource(
    "snowflake_source_for_ingest",
    query=f"select * from {source_table} order by ID limit {number_of_rows}",
    schema="schema",
    url="url",
    user="user",
    database="db",
    warehouse="warehouse",
)

feature_set = mlrun.feature_store.FeatureSet(
    "my_fs", entities=[fs.Entity("KEY")], engine="spark"
)
df = fs.ingest(
    feature_set,
    source=source,
    targets=[ParquetTarget()],
    run_config=mlrun.feature_store.RunConfig(local=False),
    spark_context=spark_context,
)

# Notice that by default, Snowflake converts to uppercase name of columns ingested to it.
# The feature-set entity, timestamp_key and label_coumnt must have similar case to the source,
# othewise the ingest will fail with MLRunInvalidArgumentError exception.
```

## Kafka source

```{admonition} Note
Support for Confluent Kafka is currently in Tech Preview status.
```

```python
profile = DatastoreProfileKafkaSource(
    name="profile-name", bootstrap_servers="localhost", topic="topic_name"
)
target = KafkaSource(path="ds://profile-name")
```

`DatastoreProfileKafkaSource` class parameters:
- `name` &mdash; Name of the profile
- `brokers` &mdash; This parameter can either be a single string or a list of strings representing the Kafka brokers. Brokers serve as the contact points for clients to connect to the Kafka cluster.
- `topics` &mdash; A string or list of strings that denote the Kafka topics from which data is sourced or read.
- `group` &mdash; A string representing the consumer group name. Consumer groups are used in Kafka to allow multiple consumers to coordinate and consume messages from topics. The default consumer group is set to `"serving"`.
- `initial_offset` &mdash; A string that defines the starting point for the Kafka consumer. It can be set to `"earliest"` to start consuming from the beginning of the topic, or `"latest"` to start consuming new messages only. The default is `"earliest"`.
- `partitions` &mdash; This can either be a single string or a list of strings representing the specific partitions from which the consumer should read. If not specified, the consumer can read from all partitions.
- `sasl_user` &mdash; A string representing the username for SASL authentication, if required by the Kafka cluster. It's tagged as private for security reasons.
- `sasl_pass` &mdash; A string representing the password for SASL authentication, correlating with the `sasl_user`. It's tagged as private for security considerations.
- `kwargs_public` &mdash; This is a dictionary (`Dict`) that holds a collection of key-value pairs used to represent settings or configurations deemed public. These pairs are subsequently passed as parameters to the underlying `kafka.KafkaProducer()` constructor. It defaults to `None`.
- `kwargs_private` &mdash; This dictionary (`Dict`) is used to store key-value pairs, typically representing configurations that are of a private or sensitive nature. These pairs are subsequently passed as parameters to the underlying `kafka.KafkaProducer()` constructor. It defaults to `None`.

See also {py:meth}`~mlrun.datastore.KafkaSource`.
  
**Example**:

```python
from mlrun.datastore.sources import KafkaSource

kafka_source = KafkaSource(
    brokers=["default-tenant.app.vmdev76.lab.iguazeng.com:9092"],
    topics="stocks-topic",
    initial_offset="earliest",
    group="my_group",
    attributes={
        "sasl": {
            "enable": True,
            "password": "pword",
            "user": "user",
            "handshake": True,
            "mechanism": "SCRAM-SHA-256",
        },
        "tls": {"enable": True, "insecureSkipVerify": False},
        "caCert": caCert,
    },
)

run_config = fstore.RunConfig(local=False).apply(mlrun.auto_mount())

stocks_set_endpoint = stocks_set.deploy_ingestion_service(
    source=kafka_source, run_config=run_config
)
```

## Parquet source
In ParquetSource, while reading a source, besides start_time and end_time,
you can also use an additional_filter attribute on other columns in your source,
which works similarly to the filtering functionality in pandas (based on pyarrow library).
This can increase performance when reading large Parquet files.

Pay attention! None/NaN/NaT values may be filtered out using this functionality on their columns.



```python
source = ParquetSource(
    "parquet_source_example",
    path="v3io://projects/example_project/source.parquet",
    time_field="hire_date",
    start_time=datetime(2023, 11, 3, 12, 30, 18),
    end_time=datetime(2023, 11, 8, 12, 30, 18),
    additional_filters=[("department", "=", "R&D")],
)
```

## S3/Azure source

When working with S3/Azure, there are additional requirements. Use: pip install mlrun[s3]; pip install mlrun[azure-blob-storage]; 
or pip install mlrun[google-cloud-storage] to install them. 
- Azure: define the environment variable `AZURE_STORAGE_CONNECTION_STRING`
- S3: define `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_BUCKET`

## SQL source

```{admonition} Note
SQL source is currently in Tech Preview status.
```
```{admonition} Limitation
Do not use SQL reserved words as entity names. See more details in [Keywords and Reserved Words](https://dev.mysql.com/doc/refman/8.0/en/keywords.html).
For currently supported versions of SQLAlchemy, see [extra-requirements.txt](https://github.com/mlrun/mlrun/blob/development/extras-requirements.txt).
See more details about [Dialects](https://docs.sqlalchemy.org/en/20/dialects/index.html).
```
{py:meth}`~mlrun.datastore.SQLSource` can be used for both batch ingestion and real time ingestion. It supports storey but does not support Spark. To configure 
either, pass the `db_url` or overwrite the `MLRUN_SQL__URL` env var, in this format:<br> 
`mysql+pymysql://<username>:<password>@<host>:<port>/<db_name>`, for example:

```python
source = SQLSource(
    table_name="my_table",
    db_path="mysql+pymysql://abc:abc@localhost:3306/my_db",
    key_field="key",
    parse_dates=["timestamp"],
)

feature_set = fs.FeatureSet(
    "my_fs",
    entities=[fs.Entity("key")],
)
feature_set.set_targets([])
df = fs.ingest(feature_set, source=source)
```

# Targets

By default, the feature sets are saved in parquet and the Iguazio NoSQL DB ({py:class}`~mlrun.datastore.NoSqlTarget`). <br>
The Parquet file is ideal for fetching large set of data for training while the key value is ideal for an online application 
since it supports low latency data retrieval based on key access. 

```{admonition} Note
When working with the Iguazio MLOps platform the default feature set storage location is under the "Projects" container: `<project name>/fs/..` folder. 
The default location can be modified in mlrun config or specified per ingest operation. The parquet/csv files can be stored in 
NFS, S3, Azure blob storage, Redis, SQL, and on Iguazio DB/FS.
```


| Class name                                    | Description                                                             | storey | spark | pandas |
| ----------------------------------------------| ------------------------------------------------------------------------| ---    | ---   | ---    |
| {py:meth}`~mlrun.datastore.CSVTarget`        |Offline. Writes events to a CSV file.                                     | Y      | Y     | Y      |
| [Kafka target](#kafka-target)                |Offline. Writes all incoming events into a Kafka stream.                  | Y      | N     | N |
| [ParquetTarget](#parquet-target)             |Offline. The Parquet target storage driver, used to materialize feature set/vector data into parquet files.| Y      | Y     | Y      |
| [SnowflakeTarget](#snowflake-target)         |Offline. Write events into tables within the Snowflake data warehouse.    | N      | Y      | N    |
| {py:meth}`~mlrun.datastore.StreamSource`     |Offline. Writes all incoming events into a V3IO stream.                   | Y      | N     | N      |
| [NoSqlTarget](#nosql-target)                 |Online. Persists the data in V3IO table to its associated storage by key. | Y      | Y     | Y      |
| [RedisNoSqlTarget](#redisnosql-target)       |Online. Persists the data in Redis table to its associated storage by key.| Y      | Y     | N      |
| [SqlTarget](#sql-target)                     |Online. Persists the data in SQL table to its associated storage by key.  | Y      | N     | Y      |


## Kafka target

```python
profile = DatastoreProfileKafkaTarget(
    name="profile-name", bootstrap_servers="localhost", topic="topic_name"
)
target = KafkaTarget(path="ds://profile-name")
```

`DatastoreProfileKafkaTarget` class parameters:
- `name` &mdash; Name of the profile
- `bootstrap_servers` &mdash; A string representing the 'bootstrap servers' for Kafka. These are the initial contact points you use to discover the full set of servers in the Kafka cluster, typically provided in the format `host1:port1,host2:port2,...`.
- `topic` &mdash; A string that denotes the Kafka topic to which data is sent or from which data is received.
- `kwargs_public` &mdash; This is a dictionary (`Dict`) meant to hold a collection of key-value pairs that could represent settings or configurations deemed public. These pairs are subsequently passed as parameters to the underlying `kafka.KafkaConsumer()` constructor. The default value for `kwargs_public` is `None`.
- `kwargs_private` &mdash; This dictionary (`Dict`) is designed to store key-value pairs, typically representing configurations that are of a private or sensitive nature. These pairs are also passed as parameters to the underlying `kafka.KafkaConsumer()` constructor. It defaults to `None`.



## Parquet target

{py:meth}`~mlrun.datastore.ParquetTarget` is the default target for offline data. 
The Parquet file is ideal for fetching large sets of data for training.

The additional_filters functionality is identical to [ParquetSource](sources-targets.md#parquet-source) behavior
while using as_df method.
### Partitioning

When writing data to a ParquetTarget, you can use partitioning. Partitioning organizes data 
in Parquet files by dividing large data sets into smaller and more manageable pieces. The data is divided
into separate files according to specific criteria, for example: date, time, or specific values in a column.
Partitioning, when configured correctly, improves read performance by reducing the amount of data that needs to be 
processed for any given function, for example, when reading back a limited time range with `get_offline_features()`.

When using the pandas engine for ingestion, pandas incurs a maximum limit of 1024 partitions on each ingestion.
If the data being ingested spans over more than 1024 partitions, the ingestion fails.
Decrease the number of partitions by filtering the time (for example, using start_filter/end_filter of the 
{py:meth}`~mlrun.datastore.ParquetSource`), and/or increasing the `time_partitioning_granularity`.

Storey processes the data row by row (as a streaming engine, it doesn't get all the data up front, so it needs to process row by row). 
These rows are batched together according to the partitions defined, and they are 
written to each partition separately. (Therefore, storey does not have the 1024 partitions limitation.)

Spark does not have the partitions limitation, either.

Configure partitioning with:
- `partitioned` &mdash; Optional. Whether to partition the file. False by default. If True without passing any other partition fields, the data is partitioned by /year/month/day/hour.
- `key_bucketing_number` &mdash; Optional. None by default: does not partition by key. 0 partitions by the key as is. Any other number "X" creates X partitions and hashes the keys to one of them.
- `partition_cols` &mdash; Optional. Name of columns from the data to partition by. 
- `time_partitioning_granularity` &mdash; Optional. The smallest time unit to partition the data by, in the format /year/month/day/hour (default). For example “hour” yields the smallest possible partitions.

For example:
- `ParquetTarget()` partitions by year/month/day/hour/
- `ParquetTarget(partition_cols=[])` writes to a directory without partitioning
- `ParquetTarget(partition_cols=["col1", "col2"])` partitions by col1/col2/
- `ParquetTarget(time_partitioning_granularity="day")` partitions by year/month/day/
- `ParquetTarget(partition_cols=["col1", "col2"], time_partitioning_granularity="day")` partitions by col1/col2/year/month/day/

Disable partitioning with:
- `ParquetTarget(partitioned=False)`

## Snowflake target

`SnowflakeTarget` parameters:
- `name`
- `user` (snowflake user)
- `warehouse` (snowflake warehouse)
- `url` (in the format: <account_name>.<region>.snowflakecomputing.com)
- `database`
- `db_schema`
- `table_name`

In addition, you need to set up this env parameter:
`SNOWFLAKE_PASSWORD`

## NoSql target

The {py:meth}`~mlrun.datastore.NoSqlTarget` is a V3IO key-value based target. It is the default target for online (real-time) data. 
It supports low latency data retrieval based on key access, making it ideal for online applications.

The combination of a NoSQL target with the storey engine does not support features of type string with a value containing both quote (') and double-quote (").

## RedisNoSql target 

```{admonition} Note
RedisNoSql target is currently in Tech Preview status.
```
See also [Redis data store profile](#redis-data-store-profile).

The Redis online target is called, in MLRun, `RedisNoSqlTarget`. The functionality of the `RedisNoSqlTarget` is identical to the `NoSqlTarget` except for:
- The RedisNoSqlTarget accepts the path parameter in the form: `<redis|rediss>://<host>[:port]`
For example: `rediss://localhost:6379` creates a Redis target, where:
   - The client/server protocol (`rediss`) is TLS protected (vs. `redis` if no TLS is established)
   - The server location is localhost port 6379.
- If the path parameter is not set, it tries to fetch it from the MLRUN_REDIS__URL environment variable.
- You cannot pass the username/password as part of the URL. If you want to provide the username/password, use secrets as:
`<prefix_>REDIS_USER <prefix_>REDIS_PASSWORD` where \<prefix> is the optional RedisNoSqlTarget `credentials_prefix` parameter.
- Two types of Redis servers are supported: StandAlone and Cluster (no need to specify the server type in the config).
- A feature set supports one online target only. Therefore `RedisNoSqlTarget` and `NoSqlTarget` cannot be used as two targets of the same feature set.
    
The K8s secrets are not available when executing locally (from the SDK). Therefore, if RedisNoSqlTarget with secret is used, 
You must add the secret as an env-var.

To use the Redis online target store, you can either change the default to be parquet and Redis, or you can specify the Redis target 
explicitly each time with the path parameter, for example:</br>
`RedisNoSqlTarget(path ="redis://1.2.3.4:6379")`

### RedisNoSql data store profile
```python
profile = DatastoreProfileRedis(
    name="profile-name",
    endpoint_url="redis://11.22.33.44:6379",
    username="user",
    password="password",
)
RedisNoSqlTarget(path="ds://profile-name/a/b")
```


## SQL target 

```{admonition} Note
Sql target is currently in Tech Preview status.
```
```{admonition} Limitation
Do not use SQL reserved words as entity names. See more details in [Keywords and Reserved Words](https://dev.mysql.com/doc/refman/8.0/en/keywords.html).
For currently supported versions of SQLAlchemy, see [extra-requirements.txt](https://github.com/mlrun/mlrun/blob/development/extras-requirements.txt).
See more details about [Dialects](https://docs.sqlalchemy.org/en/20/dialects/index.html).
```
The {py:meth}`~mlrun.datastore.SQLTarget` online target supports storey but does not support Spark. Aggregations are not supported.<br>
To configure, pass the `db_url` or overwrite the `MLRUN_SQL__URL` env var, in this format:<br>
`mysql+pymysql://<username>:<password>@<host>:<port>/<db_name>`

You can pass the schema and the name of the table you want to create or the name of an existing table, for example:

```python
target = SQLTarget(
    table_name="my_table",
    schema={"id": string, "age": int, "time": pd.Timestamp},
    create_table=True,
    primary_key_column="id",
    parse_dates=["time"],
)
feature_set = fs.FeatureSet(
    "my_fs",
    entities=[fs.Entity("id")],
)
fs.ingest(feature_set, source=df, targets=[target])
```