(sources-targets)=
# Sources and targets

- [Sources](#sources)
- [Targets](#targets)

# Sources

For batch ingestion the feature store supports dataframes and files (i.e. csv & parquet). <br>
The files can reside on S3, NFS, SQL (for example, MYSQL), Azure blob storage, or the Iguazio platform. MLRun also supports Google BigQuery as a data source. 

For real time ingestion the source can be http, Kafka, MySQL, or V3IO stream, etc.
When defining a source, it maps to nuclio event triggers. <br>

You can also create a custom `source` to access various databases or data sources.



| Class name                                                                                       | Description                                                   | storey | spark | pandas |
| --------------------------------------------------                                               | ---------------------------------                              | ---    | ---   | ---    |
| [BigQuerySource](../api/mlrun.datastore.html#mlrun.datastore.BigQuerySource)                      | Batch. Reads Google BigQuery query results as input source for a flow.| N      | Y     | Y      |
| SnowFlakeSource                                                                                 | Batch. Reads Snowflake query results as input source for a flow         | N      | Y     | N      |
| [SQLSource](#sql-data-source)                                                                    | Batch. Reads SQL query results as input source for a flow               | Y      | N     | Y      |
| [CSVSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.CSVSource)            | Batch. Reads a CSV file as input source for a flow.                   | Y      | Y     | Y      |
| [DataframeSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.DataframeSource) | Batch. Reads data frame as input source for a flow.                   | Y      | N     | N      |
| [ParquetSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.ParquetSource)    | Batch. Reads the Parquet file/dir as the input source for a flow.     | Y      | Y     | Y      |
| [HttpSource](../api/mlrun.datastore.html#mlrun.datastore.HttpSource)                               |Event-based. Sets the HTTP-endpoint source for the flow.    | Y      | N     | N      |
| [Apache Kafka source](#apache-kafka-source) and [Confluent Kafka source](#confluent-kafka-source)|Event-based. Sets the kafka source for the flow.          | Y      | N     | N      |
| [StreamSource](../api/mlrun.datastore.html#mlrun.datastore.StreamSource)                         |Event-based. Sets the stream source for the flow. If the stream doesn’t exist it creates it. | Y      | N     | N      |

## S3/Azure source

When working with S3/Azure, there are additional requirements. Use: pip install mlrun[s3]; pip install mlrun[azure-blob-storage]; 
or pip install mlrun[google-cloud-storage] to install them. 
- Azure: define the environment variable `AZURE_STORAGE_CONNECTION_STRING`
- S3: define `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_BUCKET`

## SQL source

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
                     parse_dates=['timestamp'])
 
 feature_set = fs.FeatureSet("my_fs", entities=[fs.Entity('key')],)
 feature_set.set_targets([])
 df = fs.ingest(feature_set, source=source)
```

## Apache Kafka source

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

stocks_set_endpoint = stocks_set.deploy_ingestion_service(source=kafka_source,run_config=run_config)
```

## Confluent Kafka source

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

stocks_set_endpoint = stocks_set.deploy_ingestion_service(source=kafka_source,run_config=run_config)
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

## Offline Targets

| Class name                                                                                                    | Description                                            | storey | spark | pandas |
| --------------------------------------------------                                                            | -------------------------------------------------------| ---    | ---   | ---    |
| [CSVTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.CSVTarget)        |Offline. Writes events to a CSV file.                          | Y      | Y     | Y      |
| [KafkaTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.KafkaTarget)    |Offline. Writes all incoming events into a Kafka stream.        | Y      | N     | N |
| [ParquetTarget](#parquettarget)                                                                |Offline. The Parquet target storage driver, used to materialize feature set/vector data into parquet files.                    | Y      | Y     | Y      |
| [StreamTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.StreamTarget)  |Offline. Writes all incoming events into a V3IO stream.         | Y      | N     | N      |
| [NoSqlTarget](#nosql-target)     |Online. Persists the data in V3IO table to its associated storage by key .       | Y      | Y     | Y      |
| [RedisNoSqlTarget](#redis-target) |Online. Persists the data in Redis table to its associated storage by key.                                 | Y      | Y     | N      |
| [SqlTarget](#sql-target)          |Online. Persists the data in SQL table to its associated storage by key.      | Y      | N     | Y      |


## ParquetTarget

{py:meth}`~mlrun.datastore.ParquetTarget` is the default target for offline data. 
The Parquet file is ideal for fetching large sets of data for training.

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

## NoSql target

The {py:meth}`~mlrun.datastore.NoSqlTarget` is a V3IO key-value based target. It is the default target for online (real-time) data. 
It supports low latency data retrieval based on key access, making it ideal for online applications.

The combination of a NoSQL target with the storey engine does not support features of type string with a value containing both quote (') and double-quote (").

## Redis target 

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

## SQL target 

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
            parse_dates=["time"],
        )
feature_set = fs.FeatureSet("my_fs", entities=[fs.Entity('id')],)
fs.ingest(feature_set, source=df, targets=[target])
```