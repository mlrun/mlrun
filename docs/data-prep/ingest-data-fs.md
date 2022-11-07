(ingest-data-fs)=
# Ingest data using the feature store

<!-- taken from feature-store/feature-sets -->

Define the source and material targets, and start the ingestion process (as [local process](#ingest-data-locally), [using an MLRun job](../feature-store/feature-sets.html#ingest-data-using-an-mlrun-job), [real-time ingestion](../feature-store/feature-sets.html#real-time-ingestion), or [incremental ingestion](../feature-store/feature-sets.html#incremental-ingestion)).

Data can be ingested as a batch process either by running the ingest command on demand or as a scheduled job. Batch ingestion 
can be done locally (i.e. running as a python process in the Jupyter pod) or as an MLRun job.

The data source can be a DataFrame or files (e.g. csv, parquet). Files can be either local files residing on a volume (e.g. v3io), 
or remote (e.g. S3, Azure blob). MLRun also supports Google BigQuery as a data source. If you define a transformation graph, then 
the ingestion process runs the graph transformations, infers metadata and stats, and writes the results to a target data store.

When targets are not specified, data is stored in the configured default targets (i.e. NoSQL for real-time and Parquet for offline).


```{admonition} Limitations
- Do not name columns starting with either `_` or `aggr_`. They are reserved for internal use. See 
also general limitations in [Attribute name restrictions](https://www.iguazio.com/docs/latest-release/data-layer/objects/attributes/#attribute-names).
- When using the pandas engine, do not use spaces (` `) or periods (`.`) in the column names. These cause errors in the ingestion.
```

## Inferring data

There are 2 types of infer options - metadata/schema inferring, and stats/preview inferring. The first type is responsible for describing the dataset and generating its meta-data, such as deducing the data-types of the features and listing the entities that are involved. Options belonging to this type are `Entities`, `Features` and `Index`. The `InferOptions` class has the `InferOptions.schema()` function which returns a value containing all the options of this type.
The 2nd type is related to calculating statistics and generating a preview of the actual data in the dataset. Options of this type are `Stats`, `Histogram` and `Preview`. 


The `InferOptions class` has the following values:<br>
class InferOptions:<br>
    Null = 0<br>
    Entities = 1<br>
    Features = 2<br>
    Index = 4<br>
    Stats = 8<br>
    Histogram = 16<br>
    Preview = 32<br>
    
The `InferOptions class` basically translates to a value that can be a combination of the above values. For example, passing a value of 24 means `Stats` + `Histogram`.

When simultanesouly ingesting data and requesting infer options, part of the data might be ingested twice: once for inferring metadata/stats and once for the actual ingest. This is normal behavior.

## Ingest data locally

Use a Feature Set to create the basic feature-set definition and then an ingest method to run a simple ingestion "locally" in the Jupyter Notebook pod.

```python
# Simple feature set that reads a csv file as a dataframe and ingests it as is 
stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
stocks = pd.read_csv("stocks.csv")
df = ingest(stocks_set, stocks)

# Specify a csv file as source, specify a custom CSV target 
source = CSVSource("mycsv", path="stocks.csv")
targets = [CSVTarget("mycsv", path="./new_stocks.csv")]
ingest(measurements, source, targets)
```

To learn more about ingest go to {py:class}`~mlrun.feature_store.ingest`.
