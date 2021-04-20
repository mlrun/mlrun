(transformations)=
# Feature set transformations

A feature set contains an execution graph of operations that are performed when data is ingested, or 
when simulating data flow for inferring its metadata. This graph utilizes MLRun's
[serving graph](../serving/serving-graph.md).

The graph contains steps which represent data sources and targets, and may also contain steps whose
purpose is transformations and enrichment of the data passed through the feature set. These transformations
can be provided in one of 3 ways:

* [**Aggregations**](#aggregations) - MLRun supports adding aggregate features to a feature set through the 
  {py:func}`~mlrun.feature_store.FeatureSet.add_aggregation` function.

* [**Built-in transformations**](#built-in-transformations) - MLRun is equipped with a set of transformations 
  provided through the {py:mod}`storey.flow` package. These transformations can be added to the execution graph to 
  perform common operations and transformations.
  
* [**Custom transformations**](#custom-transformations) - It is possible to extend the built-in functionality by 
  adding new classes which perform any custom operation and using them in the serving graph.

Once a feature-set was created, its internal execution graph can be observed by calling the feature-set's 
{py:func}`~mlrun.feature_store.FeatureSet.plot` function, which generates a `graphviz` plot based on the internal
graph. This is very useful when running within a Jupyter notebook, and will produce a graph such as the 
following example:

<br><img src="../_static/images/feature-store-graph.svg" alt="feature-store-graph" width="800"/><br>

This plot shows various transformations and aggregations being used as part of the feature-set processing, as well as 
the targets where results are saved to (in this case two targets). Feature-sets can also be observed in the MLRun
UI, where the full graph can be seen and specific step properties can be observed:

<br><img src="../_static/images/mlrun-ui-feature-set-graph.png" alt="ui-feature-set-graph" width="800"/><br>

For a full end-to-end example of feature-store and usage of the functionality described in this page, please refer
to the [feature store example](./feature-store-demo.ipynb).

### Aggregations

Aggregations, being a common tool in data preparation and ML feature engineering, are available directly through
the MLRun {py:class}`~mlrun.feature_store.FeatureSet` class. These transformations allow adding a new feature to the 
feature-set which is created by performing some aggregate function over feature's values within a time-based 
sliding window.

For example, if a feature-set contains stock trading data including the specific bid price for each bid at any
given time, the user may wish to introduce aggregate features which show the minimal and maximal bidding price over all 
the bids in the last hour, per stock ticker (which is the entity in question). To perform that, the following code
can be used:

```python
# create a new feature set
quotes_set = fs.FeatureSet("stock-quotes", entities=[fs.Entity("ticker")])
quotes_set.add_aggregation("bids", "bid", ["min", "max"], ["1h"], "10m")
```

Once this is executed, the feature-set will have new features introduced, with their names produced from the aggregate
parameters, using this format: `{name}_{operation}_{window}`. Thus, the example above will generate two new features:
`bid_min_1h` and `bid_max_1h`. These features can then be fed into predictive models or be used for additional 
processing and feature generation.

Aggregations which are supported using this function are:
- `sum`
- `sqr` (sum of squares)
- `max`
- `min`
- `first`
- `last`
- `avg`
- `stdvar`
- `stddev`

For a full documentation of this function, please visit the {py:func}`~mlrun.feature_store.FeatureSet.add_aggregation` 
documentation.

### Built-in transformations

MLRun, and the associated `storey` package, have a built-in library of transformation functions that can be 
applied as steps in the feature-set's internal execution graph. In order to add steps to the graph, it should be 
referenced from the {py:class}`~mlrun.feature_store.FeatureSet` object by using the 
{py:attr}`~mlrun.feature_store.FeatureSet.graph` property. Then, new steps can be added to the graph using the
functions in {py:mod}`storey.flow` (follow the link to browse the documentation and the list of existing functions).

```{admonition} Note
Internally, MLRun makes use of functions defined in the `storey` package for various purposes. When creating a 
feature-set and configuring it with sources and targets, what MLRun does behind the scenes is to add steps to the 
execution graph that wraps methods and classes which perform the actions. When defining an async execution graph,
`storey` classes are used. For example, when defining a Parquet data-target in MLRun, a graph step is created that 
wraps storey's {py:func}`~storey.writers.WriteToParquet` function.
```

To use a function, the following steps need to be taken:

1. Access the graph from the feature-set object, using the {py:attr}`~mlrun.feature_store.FeatureSet.graph` property.
2. Add steps to the graph using the various graph functions, such as {py:func}`~mlrun.feature_store.graph.to()`. 
   The function object passed to the step should point at the transformation function being used.

The following is an example for adding a simple `filter` to the graph, that will drop any bid which is lower than
50USD:

```python
quotes_set.graph.to("storey.Filter", "filter", _fn="(event['bid'] > 50)")
```

In the example above, the parameter `_fn` denotes a callable expression that will be passed ot the `storey.Filter`
class as the parameter `fn`. The callable parameter may also be a Python function, in which case there's no need for
parentheses around it. This call generates a step in the graph called `filter` which will call the expression provided
with the event being propagated through the graph as data is fed to the feature-set.

### Custom transformations

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

## Using Spark for transformations

The feature store supports using Spark for ingesting, transforming and writing results to data targets. When 
using Spark, the internal execution graph is not used, replaced by transformation code that fed to the 
{py:func}`~mlrun.feature_store.ingest` function through the following parameters:

1. A Spark session context, as the `spark_context` parameter. If the value `True` is passed here, a new Spark session
   will be created by the feature store code and will be used for execution.
   
2. Transformation runnable as the `transformer` parameter. This code will be invoked with a data-frame containing the 
   full data obtained from the source. It then should perform required transformations on the data-frame and return
   it as output.
   
The feature-store code would use Spark to read the data from the source, producing a data-frame. This data-frame will
be fed to the `transformer` provided, and then the resulting data-frame will be used to infer feature-set metadata
and it will be written using Spark to the targets. At the end of processing, the Spark session will be stopped.
