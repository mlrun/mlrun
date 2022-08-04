(ingesting_data)=
# Ingesting data

MLRun provides a set of tools and capabilities to streamline the task of data ingestion and processing. For an 
end-to-end framework for data processing, management and serving MLRun has the feature-store capabilities, which are
described [here](../feature-store/feature-store.html). However, in many cases the full feature-store capabilities are 
not needed, in which cases MLRun provides a set of utilities to facilitate data ingestion, collection and processing.

## Connecting to data sources
Accessing data from multiple source types is possible through MLRun's `DataItem` object. This object plugs into the 
data-stores framework to connect to various types of data sources and download content. For example, to download
data which is stored on S3 and load it into a `DataFrame`, use the following code:

```python
# Access object in AWS S3, in the "input-data" bucket 
import mlrun

# Access credentials
os.environ["AWS_ACCESS_KEY_ID"] = "<access key ID>"
os.environ["AWS_SECRET_ACCESS_KEY"] = "<access key>"

source_url = "s3://input-data/input_data.csv"

input_data = mlrun.get_dataitem(source_url).as_df()
```

This code runs locally (for example, in Jupyter) and relies on environment variables to supply credentials for data 
access. See [this page](../store/datastore.html) for more info on the available data-stores, accessing them locally and
remotely, and how to provide credentials for connecting.

## Data processing
Once the data is imported from its source, it can be processed using any framework. MLRun natively supports working
with Pandas DataFrames and converting from and to its `DataItem` object.

For distributed processing of very large datasets, MLRun integrates with the Spark processing engine, and provides
facilities for executing pySpark code using a Spark service (which can be deployed by the platform when running MLRun
as part of an Iguazio system) or through submitting the processing task to Spark-operator. The following pages provide
additional details and code-samples:

1. Remote spark <link?>
2. Spark operator <link?>

In a similar manner, Dask can be used for parallel processing of the data. To read data as a Dask data-frame, use the
following code:

```python
import dask.dataframe as dd

data_item = mlrun.get_dataitem(source_url)
dask_df: dd.DataFrame = data_item.as_df(df_module=dd)
```

<Should we say anything about Horovod?>
