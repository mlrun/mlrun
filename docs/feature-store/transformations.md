(transformations)=
# Feature set transformations

A feature set contains an execution graph of operations that are performed when data is ingested, or 
when simulating data flow for inferring its metadata. This graph utilizes MLRun's
[serving graph](../serving/serving-graph.md).

The graph contains steps that represent data sources and targets, and may also contain steps whose
purpose is transformations and enrichment of the data passed through the feature set. These transformations
can be provided in one of 3 ways:

* [**Aggregations**](#aggregations) - MLRun supports adding aggregate features to a feature set through the 
  {py:func}`~mlrun.feature_store.FeatureSet.add_aggregation` function.

* [**Built-in transformations**](#built-in-transformations) - MLRun is equipped with a set of transformations 
  provided through the {py:mod}`storey.transformations` package. These transformations can be added to the 
  execution graph to perform common operations and transformations.
  
* [**Custom transformations**](#custom-transformations) - It is possible to extend the built-in functionality by 
  adding new classes that perform any custom operation and using them in the serving graph.

Once a feature-set is created, its internal execution graph can be observed by calling the feature-set's 
{py:func}`~mlrun.feature_store.FeatureSet.plot` function, which generates a `graphviz` plot based on the internal
graph. This is very useful when running within a Jupyter notebook, and produces a graph such as the 
following example:

<br><img src="../_static/images/feature-store-graph.svg" alt="feature-store-graph" width="800"/><br>

This plot shows various transformations and aggregations being used as part of the feature-set processing, as well as 
the targets where results are saved to (in this case two targets). Feature-sets can also be observed in the MLRun
UI, where the full graph can be seen and specific step properties can be observed:

<br><img src="../_static/images/mlrun-ui-feature-set-graph.png" alt="ui-feature-set-graph" width="800"/><br>

For a full end-to-end example of feature-store and usage of the functionality described in this page, refer
to the [feature store example](./feature-store-demo.ipynb).

## Aggregations

Aggregations, being a common tool in data preparation and ML feature engineering, are available directly through
the MLRun {py:class}`~mlrun.feature_store.FeatureSet` class. These transformations add a new feature to the 
feature-set that is created by performing some aggregate function over the feature's values. You can use aggregation for both a time-based 
sliding window and a fixed window. In general, moving time windows are used for real time data, while fixed windows are used for historical aggregations.

- Sliding window
   
   Sliding window are fixed-size overlapping windows that slide with time. The window size can be in seconds, minutes, hours, days.
   The period determines the step size to slide. The period must be an integral divisor of the window size. 
   
   For example, a feature-set contains stock trading data including the specific bid price for each bid at any
   given time. You want to add aggregate features that show the minimal and maximal bidding price over all 
   the bids in the last 60 minutes, evaluated (sliding) at a 10 minute interval, per stock ticker (which is the entity in question). 
   To perform that, add the following code:

   ```python
   import mlrun.feature_store as fstore
   # create a new feature set
   quotes_set = fstore.FeatureSet("stock-quotes", entities=[fstore.Entity("ticker")])
   quotes_set.add_aggregation("bid", ["min", "max"], ["1h"], "10m", name="price")
   ```
   
   This code generates two new features: `bid_min_1h` and `bid_max_1h`. 
   
- Fixed window

   A fixed window has a fixed-size, is non-overlapping, and gapless. A fixed time window is used for aggregating over a time period 
   (or day of the week). For example, how busy is this restaurant between 1 and 2 pm?<br>
   When using a fixed window, each record in an in-application stream belongs to a specific window. The record is processed only once 
   (when the query processes the window to which the record belongs). Omit the period size to use a fixed window.
   
   Using the above example, but for a fixed window:
   
   ```python
   import mlrun.feature_store as fstore
   # create a new feature set
   quotes_set = fstore.FeatureSet("stock-quotes", entities=[fstore.Entity("ticker")])
   quotes_set.add_aggregation("bid", ["min", "max"], ["1h"])
   ```
   
If the `name` parameter is not specified, features are produced in the format `{column_name}_{operation}_{window}`.  
If you supply the optional `name` parameter, features are produced in the format `{name}_{operation}_{window}`.
    
These features can be fed into predictive models or be used for additional processing and feature generation.

```{admonition} Note
Internally, the graph step that is created to perform these aggregations is named `"Aggregates"`. If more than one
aggregation step is needed, supply a unique name for each, using the `step_name` parameter.
```

Aggregations that are supported using this function are:
- `count` 
- `sum`
- `sqr` (sum of squares)
- `max`
- `min`
- `first`
- `last`
- `avg`
- `stdvar`
- `stddev`

For a full documentation of this function, see the {py:func}`~mlrun.feature_store.FeatureSet.add_aggregation` 
documentation.

## Built-in transformations

MLRun, and the associated `storey` package, have a built-in library of transformation functions that can be 
applied as steps in the feature-set's internal execution graph. In order to add steps to the graph,  
reference it from the {py:class}`~mlrun.feature_store.FeatureSet` object by using the 
{py:attr}`~mlrun.feature_store.FeatureSet.graph` property. Then, new steps can be added to the graph using the
functions in {py:mod}`storey.transformations` (follow the link to browse the documentation and the 
list of existing functions). The transformations are also accessible directly from the `storey` module.

```{admonition} Note
Internally, MLRun makes use of functions defined in the `storey` package for various purposes. When creating a 
feature-set and configuring it with sources and targets, what MLRun does behind the scenes is to add steps to the 
execution graph that wraps methods and classes that perform the actions. When defining an async execution graph,
`storey` classes are used. For example, when defining a Parquet data-target in MLRun, a graph step is created that 
wraps storey's {py:func}`~storey.targets.ParquetTarget` function.
```

To use a function:

1. Access the graph from the feature-set object, using the {py:attr}`~mlrun.feature_store.FeatureSet.graph` property.
2. Add steps to the graph using the various graph functions, such as {py:func}`~mlrun.feature_store.graph.to()`. 
   The function object passed to the step should point at the transformation function being used.

The following is an example for adding a simple `filter` to the graph, that drops any bid that is lower than
50USD:

```python
quotes_set.graph.to("storey.Filter", "filter", _fn="(event['bid'] > 50)")
```

In the example above, the parameter `_fn` denotes a callable expression that is passed to the `storey.Filter`
class as the parameter `fn`. The callable parameter can also be a Python function, in which case there's no need for
parentheses around it. This call generates a step in the graph called `filter` that calls the expression provided
with the event being propagated through the graph as the data is fed to the feature-set.

## Custom transformations

When a transformation is needed that is not provided by the built-in functions, new classes that implement 
transformations can be created and added to the execution graph. Such classes should extend the 
{py:class}`~storey.flow.MapClass` class, and the actual transformation should be implemented within their `do()` 
function, which receives an event and returns the event after performing transformations and manipulations on it.
For example, consider the following code:

```python
class MyMap(MapClass):
    def __init__(self, multiplier=1, **kwargs):
        super().__init__(**kwargs)
        self._multiplier = multiplier

    def do(self, event):
        event["multi"] = event["bid"] * self._multiplier
        return event
```

The `MyMap` class can then be used to construct graph steps, in the same way as shown above for built-in functions:

```python
quotes_set.graph.add_step("MyMap", "multi", after="filter", multiplier=3)
```

This uses the `add_step` function of the graph to add a step called `multi` utilizing `MyMap` after the `filter` step 
that was added previously. The class will be initialized with a multiplier of 3.

## Using Spark execution engine

The feature store supports using Spark for ingesting, transforming and writing results to data targets. When 
using Spark, the internal execution graph is executed synchronously, by utilizing a Spark session to perform read and
write operations, as well as potential transformations on the data. Note that executing synchronously means that the 
source data is fully read into a data-frame that is processed, writing the output to the targets defined.

Spark execution can be done locally, utilizing a local Spark session provided to the ingestion call, or remotely. To 
use Spark as the transformation engine in ingestion, follow these steps:

1. When constructing the {py:class}`~mlrun.feature_store.FeatureSet` object, pass an `engine` parameter and set it 
   to `spark`. For example:
   
    ```python
    feature_set = fstore.FeatureSet("stocks", entities=[fstore.Entity("ticker")], engine="spark")
    ```

2. To use a local Spark session, pass a Spark session context when calling the 
   {py:func}`~mlrun.feature_store.ingest` function, as the `spark_context` parameter. This session is used for
   data operations and transformations.
   
3. To use a remote execution engine, pass a `RunConfig` object as the `run_config` parameter for the `ingest` API. The 
   actual remote function to execute depends on the object passed:
   
    1. A default `RunConfig`, in which case the ingestion code either generates a new MLRun function runtime
       of type `remote-spark`, or utilizes the function specified in `feature_set.spec.function` (in which case,
       it has to be of runtime type `remote-spark`).
      
    2. A `RunConfig` that has a function configured within it. As mentioned, the function runtime must be of 
       type `remote-spark`.
       
For example, the following code executes data ingestion using a local Spark session.
When using a local Spark session, the `ingest` API would wait for its completion.

```python
import mlrun
from mlrun.datastore.sources import CSVSource
import mlrun.feature_store as fstore
from pyspark.sql import SparkSession

mlrun.set_environment(project="stocks")
feature_set = fstore.FeatureSet("stocks", entities=[fstore.Entity("ticker")], engine="spark")

# add_aggregation can be used in conjunction with Spark
feature_set.add_aggregation("price", ["min", "max"], ["1h"], "10m")

source = CSVSource("mycsv", path="v3io:///projects/stocks.csv")

# Execution using a local Spark session
spark = SparkSession.builder.appName("Spark function").getOrCreate()
fstore.ingest(feature_set, source, spark_context=spark)
```

### Remote Iguazio spark ingestion example
When using remote execution the MLRun run execution details would be returned, allowing tracking of its status and results.

The following code should be executed only once to build the remote spark image before running the first ingest
It may take a few minutes to prepare the image
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
    ingest(mlrun_context=context) # The handler function must call ingest with the mlrun_context
```
You can run your PySpark code for ingesting data into the feature store by adding:
```python
def my_spark_func(df, context=None):
    return df.filter("bid>55") # PySpark code
```
```python
# mlrun: end-code
```
```python
from mlrun.datastore.sources import CSVSource
from mlrun import code_to_function
import mlrun.feature_store as fstore

feature_set = fstore.FeatureSet("stock-quotes", entities=[fstore.Entity("ticker")], engine="spark")

source = CSVSource("mycsv", path="v3io:///projects/quotes.csv")

spark_service_name = "iguazio-spark-service" # As configured & shown in the Iguazio dashboard

feature_set.graph.to(name="s1", handler="my_spark_func")
my_func = code_to_function("func", kind="remote-spark")
config = fstore.RunConfig(local=False, function=my_func, handler="ingest_handler")
fstore.ingest(feature_set, source, run_config=config, spark_context=spark_service_name)
```

### Spark execution engine over S3 - Full flow example

For Spark to work with S3, it requires several properties to be set. The following example writes a
feature set to S3 in the parquet format in a remote k8s job:

One-time setup:
1. Deploy the default image for your job (this takes several minutes but should be executed only once per cluster for any MLRun/Iguazio upgrade):
```python
from mlrun.runtimes import RemoteSparkRuntime
RemoteSparkRuntime.deploy_default_image()
```
2. Store your S3 credentials in a k8s [secret](../secrets.md):
```python
import mlrun
secrets = {'s3_access_key': AWS_ACCESS_KEY, 's3_secret_key': AWS_SECRET_KEY}
mlrun.get_run_db().create_project_secrets(
    project = "uhuh-proj",
    provider=mlrun.api.schemas.SecretProviderName.kubernetes,
    secrets=secrets
)
```

Ingestion job code (to be executed in the remote pod):
```python
# mlrun: start-code

from pyspark import SparkConf
from pyspark.sql import SparkSession


from mlrun.feature_store.api import ingest
def ingest_handler(context):
    conf = (SparkConf()
            .set("spark.hadoop.fs.s3a.path.style.access", True)
            .set("spark.hadoop.fs.s3a.access.key", context.get_secret('s3_access_key'))
            .set("spark.hadoop.fs.s3a.secret.key", context.get_secret('s3_secret_key'))
            .set("spark.hadoop.fs.s3a.endpoint", context.get_param("s3_endpoint"))
            .set("spark.hadoop.fs.s3a.region", context.get_param("s3_region"))
            .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .set("com.amazonaws.services.s3.enableV4", True)
            .set("spark.driver.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true"))
    spark = (
        SparkSession.builder.config(conf=conf).appName("S3 app").getOrCreate()
    )
    
    ingest(mlrun_context=context, spark_context=spark)
    
# mlrun: end-code
```

Ingestion invocation:
```python
from mlrun.datastore.sources import CSVSource
from mlrun import code_to_function
import mlrun.feature_store as fstore

feature_set = fstore.FeatureSet("stock-quotes", entities=[fstore.Entity("ticker")], engine="spark")

source = CSVSource("mycsv", path="v3io:///projects/quotes.csv")

spark_service_name = "iguazio-spark-service" # As configured & shown in the Iguazio dashboard

fn = code_to_function(kind='remote-spark',  name='func')

run_config=fs.RunConfig(local=False, function=fn, handler="ingest_handler")
run_config.with_secret('kubernetes', ['s3_access_key', 's3_secret_key'])
run_config.parameters = {
    "s3_endpoint" : "s3.us-east-2.amazonaws.com",
    "s3_region" : "us-east-2"
}

target = ParquetTarget(
    path = "s3://my-s3-bucket/some/path",
    partitioned = False,
)

fstore.ingest(feature_set, source, targets=[target], run_config=config, spark_context=spark_service_name)
```

