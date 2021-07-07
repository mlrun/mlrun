(datastore)=
# Data Stores & Data Items

One of the biggest challenge in distributed systems is handling data given the 
different access methods, APIs, and authentication mechanisms across types and providers.

MLRun provides 3 main abstractions to access structured and unstructured data:
* **Data Store** - defines a storage provider (e.g. file system, S3, Azure blob, Iguazio v3io, etc.)
* **Data Item** - represent a data item or collection of such (file, dir, table, etc.)
* **Artifact** - Metadata describing one or more data items. [see Artifacts](./artifacts.md).

Working with the abstractions enable us to securely access different data sources through a single API, 
many continuance methods (e.g. to/from DataFrame, get, download, list, ..), automated data movement and versioning.     

## Shared Data Stores

MLRun supports multiple data sources (more can easily added by extending the `DataStore` class)
data sources a referred to using the schema prefix (e.g. `s3://my-bucket/path`), the currently supported schemas and their urls:
* **files** - local/shared file paths, format: `/file-dir/path/to/file`
* **http, https** - read data from HTTP sources (read-only), format: `https://host/path/to/file`
* **s3** - AWS S3 objects, format: `s3://<bucket>/path/to/file`
* **v3io, v3ios** - Iguazio v3io data fabric, format: `v3io://[<remote-host>]/<data-container>/path/to/file`
* **az** - Azure Blob Store, format: `az://<bucket>/path/to/file`
* **store** - MLRun versioned artifacts [(see Artifacts)](./artifacts.md), format: `store://artifacts/<project>/<artifact-name>[:tag]`
* **memory** - in memory data registry for passing data within the same process, format `memory://key`, 
  use `mlrun.datastore.set_in_memory_item(key, value)` to register in memory data items (byte buffers or DataFrames).

Note that each data store may require connection credentials, those can be provided through function environment variables 
or project/job context secrets

## DataItem Object

When we run jobs or pipelines we pass data using the {py:class}`~mlrun.datastore.DataItem` objects, think of them as smart 
data pointers which abstract away the data store specific behavior.

Example function:

```python
def prep_data(context, source_url: mlrun.DataItem, label_column='label'):
    # Convert the DataItem to a Pandas DataFrame
    df = source_url.as_df()
    df = df.drop(label_column, axis=1).dropna()
    context.log_dataset('cleaned_data', df=df, index=False, format='csv')
```

Running our function:

```python
prep_data_run = data_prep_func.run(name='prep_data',
                                   handler=prep_data,
                                   inputs={'source_url': source_url},
                                   params={'label_column': 'userid'})
```

Note that in order to call our function with an `input` we used the `inputs` dictionary attribute and in order to pass
a simple parameter we used the `params` dictionary attribute. the input value is the specific item uri 
(per data store schema) as explained above.

The {py:class}`~mlrun.datastore.DataItem` support multiple convenience methods such as:
* **get**, **put** - to read/write data
* **download**, **upload** - to download/upload files
* **as_df** - to convert the data to a DataFrame object
* **local** - to get a local file link to the data (will be downloaded locally if needed)
* **listdir**, **stat** - file system like methods
* **meta** - access to the artifact metadata (in case of an artifact uri)

Check the **{py:class}`~mlrun.datastore.DataItem`** class documentation for details

In order to get a DataItem object from a url use {py:func}`~mlrun.run.get_data_item` or 
{py:func}`~mlrun.run.get_data_object` (returns the `DataItem.get()`), for example:

    df = mlrun.get_data_item('s3://demo-data/mydata.csv').as_df()
    print(mlrun.get_data_object('https://my-site/data.json'))

