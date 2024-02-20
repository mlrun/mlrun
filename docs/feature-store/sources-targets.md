(sources-targets)=
# Sources and targets


- [Sources](#sources)
- [Targets](#targets)
- [ParquetTarget](#parquettarget)
- [NoSql target](#nosql-target)



# Sources
| Class name                                                                                                        | Description                                                                     | storey | spark | pandas |
| --------------------------------------------------                                                                | ---------------------------------                                               | ---    | ---   | ---    |
| [mlrun.datastore.BigQuerySource](../api/mlrun.datastore.html#mlrun.datastore.BigQuerySource)                      | Reads Google BigQuery query results as input source for a flow ("batch" source).                 | N      | Y     | Y      |
| mlrun.datastore.SnowFlakeSource                                                                                 | Reads Snowflake query results as input source for a flow ("batch" source).                         | N      | Y     | N      |
| mlrun.datastore.SQLSource                                                                                       | Reads SQL query results as input source for a flow ("batch" source).                               | Y      | N     | Y      |
| [mlrun.datastore.CSVSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.CSVSource)            | Reads a CSV file as input source for a flow ("batch" source).                                    | Y      | Y     | Y      |
| [storey.sources.DataframeSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.DataframeSource) | Reads data frame as input source for a flow ("batch" source).                                    | Y      | N     | N      |
| [mlrun.datastore.HttpSource](../api/mlrun.datastore.html#mlrun.datastore.HttpSource)                              | Sets the HTTP-endpoint source for the flow (event-based source).                                 | Y      | N     | N      |
| [mlrun.datastore.KafkaSource](../api/mlrun.datastore.html#mlrun.datastore.KafkaSource)                            | Sets the kafka source for the flow (event-based source).                                         | Y      | N     | N      |
| [mlrun.datastore.ParquetSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.ParquetSource)    | Reads the Parquet file/dir as the input source for a flow ("batch" source).                      | Y      | Y     | Y      |
| [mlrun.datastore.StreamSource](../api/mlrun.datastore.html#mlrun.datastore.StreamSource)                          | Sets the stream source for the flow. If the stream doesn’t exist it creates it (event-based source). | Y      | N     | N      |

# Targets
| Class name                                                                                                    | Description                                                                                        | storey | spark | pandas |
| --------------------------------------------------                                                            | -------------------------------------------------------                                            | ---    | ---   | ---    |
| [mlrun.datastore.CSVTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.CSVTarget)        | Writes events to a CSV file (offline target).                                                                       | Y      | Y     | Y      |
| [mlrun.datastore.KafkaTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.KafkaTarget)    | Writes all incoming events into a Kafka stream (online target).                                                    | Y | N | N |
| [mlrun.datastore.NoSqlTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.NoSqlTarget)    | The default online target. Persists the data in V3IO table to its associated storage by key (online target).       | Y      | Y     | Y      |
| mlrun.datastore.RedisNoSqlTarget                                                                              | Persists the data in Redis table to its associated storage by key (online target).                                 | Y      | Y     | N      |
| mlrun.datastore.SqlTarget                                                                                     | The default offline target. Persists the data in SQL table to its associated storage by key (offline target).      | Y      | N     | Y      |
| [mlrun.datastore.ParquetTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.ParquetTarget)| The Parquet target storage driver, used to materialize feature set/vector data into parquet files (online target). | Y      | Y     | Y      |
| [mlrun.datastore.StreamTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.StreamTarget)  | Writes all incoming events into a V3IO stream (offline target).                                                    | Y      | N     | N      |

## ParquetTarget

{py:meth}`~mlrun.datastore.ParquetTarget` is the default target for offline data. 
The Parquet file is ideal for fetching large set of data for training.

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

The {py:meth}`~mlrun.datastore.NoSqlTarget` is a V3IO key-value based target. It is the default target for real-time data. 
It supports low latency data retrieval based on key access, making it ideal for online applications.

The combination of a NoSQL target with the storey engine does not support features of type string with a value containing both quote (') and double-quote (").
