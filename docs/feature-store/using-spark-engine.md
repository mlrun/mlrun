(ingest-features-spark)=
# Ingest features with Spark

The feature store supports using Spark for ingesting, transforming, and writing results to data targets. When 
using Spark, the internal execution graph is executed synchronously by utilizing a Spark session to perform read and
write operations, as well as potential transformations on the data. Executing synchronously means that the 
source data is fully read into a data-frame that is processed, writing the output to the targets defined.

To use Spark as the transformation engine in ingestion, follow these steps:

When constructing the {py:class}`~mlrun.feature_store.FeatureSet` object, pass an `engine` parameter and set it 
   to `spark`. For example:
   
```python
feature_set = fstore.FeatureSet(
    "stocks", entities=[fstore.Entity("ticker")], engine="spark"
)
```
To use a remote execution engine, pass a `RunConfig` object as the `run_config` parameter for 
the `ingest` API. The actual remote function to execute depends on the object passed:
   
   - A default `RunConfig`, in which case the ingestion code either generates a new MLRun function runtime
       of type `remote-spark`, or utilizes the function specified in `feature_set.spec.function` (in which case,
       it has to be of runtime type `remote-spark` or `spark`).
      
   - A `RunConfig` that has a function configured within it. As mentioned, the function runtime must be of 
       type `remote-spark` or `spark`.
       
Spark execution can be done locally, utilizing a local Spark session provided to the ingestion call. To use a local Spark session, pass a 
Spark session context when calling the {py:func}`~mlrun.feature_store.ingest` function, as the 
`spark_context` parameter. This session is used for data operations and transformations.
       
See code examples in:
- [Local Spark ingestion example](#local-spark-ingestion-example)
- [Remote Spark ingestion example](#remote-spark-ingestion-example)
- [Spark operator ingestion example](#spark-operator-ingestion-example)
- [Spark dataframe ingestion example](#spark-dataframe-ingestion-example)
- [Spark over S3 full flow example](#spark-over-s3-full-flow-example)
- [Spark ingestion from Snowflake example](#spark-ingestion-from-snowflake-example)
- [Spark ingestion from Azure example](#spark-ingestion-from-azure-example)
       
## Local Spark ingestion example

A local Spark session is a session running in the Jupyter service.<br>
The following code executes data ingestion using a local Spark session. 

When using a local Spark session, the `ingest` API would wait for its completion. 

```python
import mlrun
from mlrun.datastore.sources import CSVSource
import mlrun.feature_store as fstore
from pyspark.sql import SparkSession

mlrun.get_or_create_project(name="stocks")
feature_set = fstore.FeatureSet(
    "stocks", entities=[fstore.Entity("ticker")], engine="spark"
)

# add_aggregation can be used in conjunction with Spark
feature_set.add_aggregation("price", ["min", "max"], ["1h"], "10m")

source = CSVSource("mycsv", path="v3io:///projects/stocks.csv")

# Execution using a local Spark session
spark = SparkSession.builder.appName("Spark function").getOrCreate()
feature_set.ingest(source, spark_context=spark)
```

## Remote Spark ingestion example

Remote Spark refers to  a session running from another service, for example, the Spark standalone service or the Spark operator service. 
When using remote execution the MLRun run execution details are returned, allowing tracking of its status and results. 

The following code should be executed only once to build the remote spark image before running the first ingest.
It may take a few minutes to prepare the image.

```python
from mlrun.runtimes import RemoteSparkRuntime

RemoteSparkRuntime.deploy_default_image()
```

Remote ingestion:
```python
# mlrun: start-code
```
```python
from mlrun.feature_store.api import ingest


def ingest_handler(context):
    ingest(
        mlrun_context=context
    )  # The handler function must call ingest with the mlrun_context
```
You can run your PySpark code for ingesting data into the feature store by adding:
```python
def my_spark_func(df, context=None):
    return df.filter("bid>55")  # PySpark code
```
```python
# mlrun: end-code
```
```python
from mlrun.datastore.sources import CSVSource
import mlrun.feature_store as fstore

mlrun.get_or_create_project(name="remote-spark")

feature_set = fstore.FeatureSet(
    "stock-quotes", entities=[fstore.Entity("ticker")], engine="spark"
)

source = CSVSource("mycsv", path="v3io:///projects/quotes.csv")

spark_service_name = (
    "iguazio-spark-service"  # As configured & shown in the Iguazio dashboard
)

feature_set.graph.to(name="s1", handler="my_spark_func")
my_func = project.set_function("func", kind="remote-spark")
config = fstore.RunConfig(local=False, function=my_func, handler="ingest_handler")
feature_set.ingest(source, run_config=config, spark_context=spark_service_name)
```

## Spark operator ingestion example
When running with a Spark operator, the MLRun execution details are returned, allowing tracking of the job's status and results. Spark operator ingestion is always executed remotely.

The following code should be executed only once to build the spark job image before running the first ingest.
It may take a few minutes to prepare the image.
```python
from mlrun.runtimes import Spark3Runtime

Spark3Runtime.deploy_default_image()
```

Spark operator ingestion:
```python
# mlrun: start-code

from mlrun.feature_store.api import ingest


def ingest_handler(context):
    ingest(
        mlrun_context=context
    )  # The handler function must call ingest with the mlrun_context


# You can add your own PySpark code as a graph step:
def my_spark_func(df, context=None):
    return df.filter("bid>55")  # PySpark code


# mlrun: end-code
```

```python
from mlrun.datastore.sources import CSVSource
import mlrun.feature_store as fstore

mlrun.get_or_create_project(name="spark-oper")

feature_set = fstore.FeatureSet(
    "stock-quotes", entities=[fstore.Entity("ticker")], engine="spark"
)

source = CSVSource("mycsv", path="v3io:///projects/quotes.csv")

feature_set.graph.to(name="s1", handler="my_spark_func")

my_func = project.set_function("func", kind="spark")

my_func.with_driver_requests(cpu="200m", mem="1G")
my_func.with_executor_requests(cpu="200m", mem="1G")
my_func.with_igz_spark()

# Enables using the default image (can be replace with specifying a specific image with .spec.image)
my_func.spec.use_default_image = True

# Not a must - default: 1
my_func.spec.replicas = 2

# If needed, sparkConf can be modified like this:
# my_func.spec.spark_conf['spark.specific.config.key'] = 'value'

config = fstore.RunConfig(local=False, function=my_func, handler="ingest_handler")
feature_set.ingest(source, run_config=config)
```



## Spark dataframe ingestion example

The following code executes local data ingestion from a spark dataframe (Spark dataframe Ingestion cannot be executed remotely.)
The specified dataframe should be associated with `spark_context`. 

```
from pyspark.sql import SparkSession
import mlrun.feature_store as fstore

columns = ["id", "count"]
data = [("a", "12"), ("b", "14"), ("c", "88")]

spark = SparkSession.builder.appName('example').getOrCreate()
df = spark.createDataFrame(data).toDF(*columns)

fset = fstore.FeatureSet("myset", entities=[fstore.Entity("id")], engine="spark")

fset.ingest(df, spark_context=spark)

spark.stop()
```

## Spark over S3 - full flow example

For Spark to work with S3, it requires several properties to be set. Spark over S3 can be executed both remotely and locally, as long as access credentials to the S3 
objects are available to it. The following example writes a
feature set to S3 in the parquet format in a remote k8s job:

One-time setup:
1. Deploy the default image for your job (this takes several minutes but should be executed only once per cluster for any MLRun/Iguazio upgrade):
   ```python
   from mlrun.runtimes import RemoteSparkRuntime

   RemoteSparkRuntime.deploy_default_image()
   ```
2. Store your S3 credentials in a k8s [secret](../secrets.html#kubernetes-project-secrets):
   ```python
   import mlrun

   secrets = {"s3_access_key": AWS_ACCESS_KEY, "s3_secret_key": AWS_SECRET_KEY}
   mlrun.get_run_db().create_project_secrets(
       project="uhuh-proj",
       provider=mlrun.common.schemas.SecretProviderName.kubernetes,
       secrets=secrets,
   )
   ```

Ingestion job code (to be executed in the remote pod):
```python
# mlrun: start-code

from pyspark import SparkConf
from pyspark.sql import SparkSession


from mlrun.feature_store.api import ingest


def ingest_handler(context):
    conf = (
        SparkConf()
        .set("spark.hadoop.fs.s3a.path.style.access", True)
        .set("spark.hadoop.fs.s3a.access.key", context.get_secret("s3_access_key"))
        .set("spark.hadoop.fs.s3a.secret.key", context.get_secret("s3_secret_key"))
        .set("spark.hadoop.fs.s3a.endpoint", context.get_param("s3_endpoint"))
        .set("spark.hadoop.fs.s3a.region", context.get_param("s3_region"))
        .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .set("com.amazonaws.services.s3.enableV4", True)
        .set(
            "spark.driver.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true"
        )
    )
    spark = SparkSession.builder.config(conf=conf).appName("S3 app").getOrCreate()

    ingest(mlrun_context=context, spark_context=spark)


# mlrun: end-code
```

Ingestion invocation:
```python
from mlrun.datastore.sources import CSVSource
from mlrun.datastore.targets import ParquetTarget
import mlrun.feature_store as fstore

feature_set = fstore.FeatureSet(
    "stock-quotes", entities=[fstore.Entity("ticker")], engine="spark"
)

source = CSVSource("mycsv", path="v3io:///projects/quotes.csv")

spark_service_name = "spark"  # As configured & shown in the Iguazio dashboard

fn = project.set_function(kind="remote-spark", name="func")

run_config = fstore.RunConfig(local=False, function=fn, handler="ingest_handler")
run_config.with_secret("kubernetes", ["s3_access_key", "s3_secret_key"])
run_config.parameters = {
    "s3_endpoint": "s3.us-east-2.amazonaws.com",
    "s3_region": "us-east-2",
}

target = ParquetTarget(
    path="s3://my-s3-bucket/some/path",
    partitioned=False,
)

feature_set.ingest(
    source, targets=[target], run_config=run_config, spark_context=spark_service_name
)
```

## Spark ingestion from Snowflake example

Spark ingestion from Snowflake can be executed both remotely and locally. 

When running aggregations, they actually run on Spark and require Spark compute resources.<br>
The queries from the database are "regular" snowflake queries and they use Snowflake compute resources.

```{admonition} Note
`Entity` is case sensitive.
```

The following code executes local data ingestion from Snowflake.

```
from pyspark.sql import SparkSession

import mlrun
import mlrun.feature_store as fstore
from mlrun.datastore.sources import SnowflakeSource

spark = SparkSession.builder.appName("snowy").getOrCreate()

mlrun.get_or_create_project("feature_store")
feature_set = fstore.FeatureSet(
    name="customer", entities=[fstore.Entity("C_CUSTKEY")], engine="spark"
)

source = SnowflakeSource(
    "customer_sf",
    query="select * from customer limit 100000",
    url="<url>",
    user="<user>",
    password="<password>",
    database="SNOWFLAKE_SAMPLE_DATA",
    db_schema="TPCH_SF1",
    warehouse="compute_wh",
)

feature_set.ingest(source, spark_context=spark)
```

## Spark ingestion from Azure example

Spark ingestion from Azure can be executed both remotely and locally. The following code executes remote data ingestion from Azure.

```
import mlrun

# Initialize the MLRun project object
project_name = "spark-azure-test"
project = mlrun.get_or_create_project(project_name, context="./")

from mlrun.runtimes import RemoteSparkRuntime
RemoteSparkRuntime.deploy_default_image()

from mlrun.datastore.sources import CSVSource
from mlrun.datastore.targets import ParquetTarget
import mlrun.feature_store as fstore

feature_set = fstore.FeatureSet("rides7", entities=[fstore.Entity("ride_id")], engine="spark", timestamp_key="key")

source = CSVSource("rides", path="wasbs://warroom@mlrunwarroom.blob.core.windows.net/ny_taxi_train_subset_ride_id.csv")

spark_service_name = "spark-fs" # As configured & shown in the Iguazio dashboard

fn = project.set_function(kind='remote-spark',  name='func')

run_config = fstore.RunConfig(local=False, function=fn, handler="ingest_handler")

target = ParquetTarget(partitioned = True, time_partitioning_granularity="month")

feature_set.set_targets(targets=[target],with_defaults=False)

feature_set.ingest(source, run_config=run_config, spark_context=spark_service_name)
```