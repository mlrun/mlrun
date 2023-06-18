(feature-sets)=
# Feature sets

In MLRun, a group of features can be ingested together and stored in logical group called feature set. 
Feature sets take data from offline or online sources, build a list of features through a set of transformations, and 
store the resulting features along with the associated metadata and statistics. <br>
A feature set can be viewed as a database table with multiple material implementations for batch and real-time access,
along with the data pipeline definitions used to produce the features.
 
The feature set object contains the following information:
- **Metadata** &mdash; General information which is helpful for search and organization. Examples are project, name, owner, last update, description, labels, etc.
- **Key attributes** &mdash; Entity, timestamp key (optional), label column.
- **Features** &mdash; The list of features along with their schema, metadata, validation policies and statistics.
- **Source** &mdash; The online or offline data source definitions and ingestion policy (file, database, stream, http endpoint, etc.). See the [source descriptions](../serving/available-steps.html#sources).
- **Transformation** &mdash; The data transformation pipeline (e.g. aggregation, enrichment etc.).
- **Target stores** &mdash; The type (i.e. parquet/csv or key value), location and status for the feature set materialized data. See the [target descriptions](../serving/available-steps.html#targets).
- **Function** &mdash; The type (storey, pandas, spark) and attributes of the data pipeline serverless functions.

**In this section**
- [Create a Feature Set](#create-a-feature-set)
- [Create a feature set without ingesting its data](#create-a-feature-set-without-ingesting-its-data)
- [Add transformations](#add-transformations)

**See also**:
- [Verify a feature set with a small dataset by inferring data](../data-prep/ingest-data-fs.html#verify-a-feature-set-with-a-small-dataset-by-inferring-data)
- {ref}`Ingest data using the feature store <ingest-data-fs>`


   
## Create a feature set

Create a {py:class}`~mlrun.feature_store.FeatureSet` with the base definitions:

* **name** &mdash; The feature set name is a unique name within a project. 
* **entities** &mdash; Each feature set must be associated with one or more index column. When joining feature sets, the key columns 
   are determined by the relations field if it exists, and otherwise by the entities.
* **timestamp_key** &mdash; (optional) Used for specifying the time field when joining by time.
* **engine** &mdash; The processing engine type:
   - Spark
   - pandas
   - storey. Default. (Some advanced functionalities are in the Beta state.)
* **label_column** &mdash; Name of the label column (the one holding the target (y) values).
* **relations** &mdash; (optional) Dictionary that indicates all of the relations between current feature set to other featuresets . It looks like: `{"<my_column_name>":Entity, ...}`. If the feature_set relations is None, the join is done based on feature_set entities. Relevant only for Dask and storey (local) engines.
   See more about joins in [Using joins in an offline feature vector](./feature-vectors.html#using-joins-in-an-offline-feature-vector). 
   
Example:
```python
#Create a basic feature set example
stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
```


### Create a feature set in the UI

1. Select a project and press **Feature store**, then press **Create Set**.
3. After completing the form, press **Save and Ingest** to start the process, or **Save** to save the set for later ingestion.

## Create a feature set without ingesting its data

You can define and register a feature set (and use it in a feature vector) without ingesting its data into MLRun offline targets. This supports all batch sources.

The use-case for this is when you have a large amount of data in a remote storage that is ready to be consumed by a model-training pipeline.
When this feature is enabled on a feature set, data is **not** saved to the offline target during ingestion. Instead, when `get_offline_features` 
is called on a vector containing that feature set, that data is read directly from the source.
Online targets are still ingested, and their value represents a timeslice of the offline source.
Transformations are not allowed when this feature is enabled: no computation graph, no aggregations, etc.
Enable this feature by including `passthrough=True` in the feature set definition. All three ingestion engines (Storey, Spark, Pandas) 
are supported, as well as the retrieval engines "local" and "spark".

Typical code, from defining the feature set through ingesting its data:
```
# Flag the feature set as passthrough
my_fset = fstore.FeatureSet("my_fset", entities=[Entity("patient_id)], timestamp_key="timestamp", passthrough=True) 
csv_source = CSVSource("my_csv", path="data.csv"), time_field="timestamp")
# Ingest the source data, but only to online/nosql target
fstore.ingest(my_fset, csv_source) 
vector = fstore.FeatureVector("myvector", features=[f"my_fset"])
# Read the offline data directly from the csv source
resp = fstore.get_offline_features(vector, entity_timestamp_column="timestamp", with_indexes=True) 
```


## Add transformations 

Define the data processing steps using a transformations graph (DAG).

A feature set data pipeline takes raw data from online or offline sources and transforms it to meaningful features.
The MLRun feature store supports three processing engines (storey, pandas, spark) that can run in the client 
(e.g. Notebook) for interactive development or in elastic serverless functions for production and scale.

The data pipeline is defined using MLRun graph (DAG) language. Graph steps can be pre-defined operators 
(such as aggregate, filter, encode, map, join, impute, etc.) or custom python classes/functions. 
Read more about the graph in [Real-time serving pipelines (graphs)](../serving/serving-graph.html).

The `pandas` and `spark` engines are good for simple batch transformations, while the `storey` stream processing engine (the default engine)
can handle complex workflows and real-time sources.

The results from the transformation pipeline are stored in one or more material targets.  Data for offline 
access, such as training, is usually stored in Parquet files. Data for online access such as serving is stored 
in the Iguazio NoSQL DB (`NoSqlTarget`). You can use the default targets or add/replace with additional custom targets.

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

See more details in {ref}`Feature set transformations <transformations>`.

## Simulate and debug the data pipeline with a small dataset
During the development phase it's pretty common to check the feature set definition and to simulate the creation of the feature set before 
ingesting the entire dataset, since ingesting the entire feature set can take time. <br>
This allows you to get a preview of the results (in the returned dataframe). The simulation method is called `preview`. It previews in the source 
data schema, as well as processing the graph logic (assuming there is one) on a small subset of data. 
The preview operation also learns the feature set schema and does statistical analysis on the result by default.
  
```python
df = fstore.preview(quotes_set, quotes)

# print the featue statistics
print(quotes_set.get_stats_table())
```

