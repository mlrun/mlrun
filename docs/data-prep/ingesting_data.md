(ingesting_data)=
# Using data sources and items

**In this section**
- [Connecting to data sources](#connecting-to-data-sources)
- [Data processing](#data-processing)

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
access. See {ref}`datastore` for more info on the available data-stores, accessing them locally and
remotely, and how to provide credentials for connecting. 

Running the code locally is very useful for easy debugging and development of the code. 
When the code moves to a stable status, it is usually recommended to run it "remotely" on a pod running in the 
Kubernetes cluster. This allows setting up specific resources to the processing pod 
(such as memory, CPU and execution priority).

MLRun provides facilities to create `DataItem` objects as inputs to running code. For example, this is a basic
data ingestion function:

```python
def ingest_data(context, source_url: mlrun.DataItem):
    # Load the data from its source, and convert to a DataFrame
    df = source_url.as_df()

    # Perform data cleaning and processing
    # ...

    # Save the processed data to the artifact store
    context.log_dataset("cleaned_data", df=df, format="csv")
```

This code can be placed in a python file, or as a cell in the Python notebook. For example, if the code above was saved
to a file, the following code creates an MLRun function from it and executes it remotely in a pod:

```python
# create a project, then a function from py or notebook (ipynb) file, specify the default function handler
project = mlrun.get_or_create_project("ingest-data")
ingest_func = project.set_function(
    name="ingest_data", filename="./ingest_data.py", kind="job", image="mlrun/mlrun"
)

source_url = "s3://input-data/input_data.csv"

ingest_data_run = ingest_func.run(
    name="ingest_data",
    handler=ingest_data,
    inputs={"source_url": source_url},
    local=False,
)
```

As the `source_url` is part of the function's `inputs`, MLRun automatically wraps it up with a `DataItem`. The output
is logged to the function's `artifact_path`, and can be obtained from the run result:

```python
cleaned_data_frame = ingest_data_run.artifact("cleaned_data").as_df()
```

Note that running the function remotely may require attaching storage to the function, as well as passing storage
credentials through project secrets. See the following pages for more details:

1. {ref}`Function_storage_auto_mount`
2. {ref}`secrets`

## Data processing
Once the data is imported from its source, it can be processed using any framework. MLRun natively supports working
with Pandas DataFrames and converting from and to its `DataItem` object.

For distributed processing of very large datasets, MLRun integrates with the Spark processing engine, and provides
facilities for executing pySpark code using a Spark service (which can be deployed by the platform when running MLRun
as part of an Iguazio system) or through submitting the processing task to Spark-operator. The following page provides
additional details and code-samples:

- [Spark operator](../runtimes/spark-operator.html)

In a similar manner, Dask can be used for parallel processing of the data. To read data as a Dask `DataFrame`, use the
following code:

```python
import dask.dataframe as dd

data_item = mlrun.get_dataitem(source_url)
dask_df: dd.DataFrame = data_item.as_df(df_module=dd)
```

