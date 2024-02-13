(sources-targets)=
# Sources and targets


- [Sources](#sources)
- [Targets](#targets)
- [Partitioning on Parquet target](#partitioning-on-parquet-target)



## Sources
| Class name                                                                                                        | Description                                                                     | storey | spark | pandas |batch or event|
| --------------------------------------------------                                                                | ---------------------------------                                               | ---    | ---   | ---    |---|
| [mlrun.datastore.BigQuerySource](../api/mlrun.datastore.html#mlrun.datastore.BigQuerySource)                      | Reads Google BigQuery query results as input source for a flow.                 | N      | Y     | Y      |batch|
| mlrun.datastore.SnowFlakeSource                                                                                 | Reads Snowflake query results as input source for a flow.                         | N      | Y     | N      |batch|
| mlrun.datastore.SQLSource                                                                                       | Reads SQL query results as input source for a flow.                               | Y      | N     | Y      |batch|
| [mlrun.datastore.CSVSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.CSVSource)            | Reads a CSV file as input source for a flow.                                    | Y      | Y     | Y      |batch|
| [storey.sources.DataframeSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.DataframeSource) | Reads data frame as input source for a flow.                                    | Y      | N     | N      |batch|
| [mlrun.datastore.HttpSource](../api/mlrun.datastore.html#mlrun.datastore.HttpSource)                              | Sets the HTTP-endpoint source for the flow.                                     | Y      | N     | N      |event|
| [mlrun.datastore.KafkaSource](../api/mlrun.datastore.html#mlrun.datastore.KafkaSource)                            | Sets the kafka source for the flow.                                             | Y      | N     | N      |event|
| [mlrun.datastore.ParquetSource](https://storey.readthedocs.io/en/latest/api.html#storey.sources.ParquetSource)    | Reads the Parquet file/dir as the input source for a flow.                      | Y      | Y     | Y      |batch|
| [mlrun.datastore.StreamSource](../api/mlrun.datastore.html#mlrun.datastore.StreamSource)                          | Sets the stream source for the flow. If the stream doesn’t exist it creates it. | Y      | N     | N      |event|

## Targets
| Class name                                                                                                    | Description                                                                                        | storey | spark | pandas |Online or offline|
| --------------------------------------------------                                                            | -------------------------------------------------------                                            | ---    | ---   | ---    |---|
| [mlrun.datastore.CSVTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.CSVTarget)        | Writes events to a CSV file.                                                                       | Y      | Y     | Y      |offline|
| [mlrun.datastore.KafkaTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.KafkaTarget)    | Writes all incoming events into a Kafka stream.                                                    |  |  |  |  offline|
| [mlrun.datastore.NoSqlTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.NoSqlTarget)    | Persists the data in V3IO table to its associated storage by key.                                  | Y      | Y     | Y      |Online (default)|
| mlrun.datastore.RedisNoSqlTarget                                                                              | Persists the data in Redis table to its associated storage by key.                                 | Y      | Y     | N      |Online|
| mlrun.datastore.SqlTarget                                                                                     | Persists the data in SQL table to its associated storage by key.                                   | Y      | N     | Y      |Offline (default)|
| [mlrun.datastore.ParquetTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.ParquetTarget)| The Parquet target storage driver, used to materialize feature set/vector data into parquet files. | Y      | Y     | Y      |Online|
| [mlrun.datastore.StreamTarget](https://storey.readthedocs.io/en/latest/api.html#storey.targets.StreamTarget)  | Writes all incoming events into a V3IO stream.                                                     | Y      | N     | N      |Offline|

## Partitioning on Parquet target

Partitioning organizes data in Parquet files by dividing large data sets into smaller and more manageable pieces. The data is divided
into separate files according to specific criteria, for example: date, time, or specific values in a column.
Partitioning, when configured correctly, improves read performance by reducing the amount of data that needs to be procesed for any function.
Partitioning is supported for {py:meth}`~mlrun.datastore.ParquetTarget`. Configure partitioning with:

- `partitioned` &mdash; Optional. Whether to partition the file. False by default. Ff True without passing any other partition fields, the data is partitioned by /year/month/day/hour.
- `key_bucketing_number` &mdash; Optional. None by default: does not partition by key. 0 partitions by the key as is. Any other number "X" creates X partitions and hashes the keys to one of them.
- `partition_cols` &mdash; Optional. Name of columns from the data to partition by.
- `time_partitioning_granularity` &mdash; Optional. The smallest time unit to partition the data by. For example “hour” yields the smallest possible partitions, in the format /year/month/day/hour.


