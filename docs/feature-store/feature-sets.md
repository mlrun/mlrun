(feature-sets)=
# Feature sets

In MLRun, a group of features can be ingested together and stored in logical group called feature set. 
Feature sets take data from offline or online sources, build a list of features through a set of transformations, and 
store the resulting features along with the associated metadata and statistics. <br>
A feature set can be viewed as a database table with multiple material implementations for batch and real-time access,
along with the data pipeline definitions used to produce the features.
 
The feature set object contains the following information:
- **Metadata** &mdash; General information which is helpful for search and organization. Examples are project, name, owner, last update, description, labels, etc.
- **Key attributes** &mdash; Entity (the join key), timestamp key (optional), label column.
- **Features** &mdash; The list of features along with their schema, metadata, validation policies and statistics.
- **Source** &mdash; The online or offline data source definitions and ingestion policy (file, database, stream, http endpoint, etc.).
- **Transformation** &mdash; The data transformation pipeline (e.g. aggregation, enrichment etc.).
- **Target stores** &mdash; The type (i.e. parquet/csv or key value), location and status for the feature set materialized data. 
- **Function** &mdash; The type (storey, pandas, spark) and attributes of the data pipeline serverless functions.

**In this section**
- [Create a Feature Set](#create-a-feature-set)
- [Add transformations](#add-transformations)
- [Simulate and debug the data pipeline with a small dataset](#simulate-the-data-pipeline-with-a-small-dataset)


See also [Ingest data using the feature store](#ingest-data-fs)
  
   
## Create a feature set

Create a {py:class}`~mlrun.feature_store.FeatureSet` with the base definitions:

* **name** &mdash; The feature set name is a unique name within a project. 
* **entities** &mdash; Each feature set must be associated with one or more index column. When joining feature sets, the key columns 
   are determined by the the relations field if it exists, and otherwise by the entities.
* **timestamp_key** &mdash; (optional) Used for specifying the time field when joining by time.
* **engine** &mdash; The processing engine type:
   - Spark
   - pandas
   - storey. Default. (Some advanced functionalities are in the Beta state.)
* **label_column** &mdash; Name of the label column (the one holding the target (y) values).
* **relations** &mdash; (optional) Dictionary that indicates all of the relations between different feature sets. It looks like: `{"feature_set_name_1:feature_set_name_2":{"column_of_1":"column_of_2",...}...}`. If the relation is None, and the `feature_set` 
   relations is also None, the join is done on the entity. Relevant only for Dask and storey (local) engines.<br>
   You can define the relations of a feature set with the relations argument, like this:
   `{"feature_set_name": {"my_column":"other_feature_set_column", ...}...}`<br>
   See more about joins in [Creating an offline feature vector](./feature-vectors.html#creating-an-offline-feature-vector).
   
Example:
```python
#Create a basic feature set example
stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
```

To learn more about Feature Sets go to {py:class}`~mlrun.feature_store.FeatureSet`.

```{admonition} Note 
Feature sets can also be created in the UI. To create a feature set:
1. Select a project and press **Feature store**, then press **Create Set**.
3. After completing the form, press **Save and Ingest** to start the process, or **Save** to save the set for later ingestion.
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
in the Iguazio NoSQL DB (` NoSqlTarget`). You can use the default targets or add/replace with additional custom targets. See Target stores(#target-stores).

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

See more details in [Feature set transformations](transformations.html).

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