(data-items)=
# Data items

A data item can be one item or a collection of items (file, dir, table, etc.).

When running jobs or pipelines, data is passed using the {py:class}`~mlrun.datastore.DataItem` objects. Data items objects abstract away
the data backend implementation, provide a set of convenience methods (`.as_df`, `.get`, `.show`, ..), and enable auto logging/versioning
of data and metadata.

Example function:

```python
# Save this code as a .py file:
import mlrun

def prep_data(context, source_url: mlrun.DataItem, label_column='label'):
    # Convert the DataItem to a Pandas DataFrame
    df = source_url.as_df()
    df = df.drop(label_column, axis=1).dropna()
    context.log_dataset('cleaned_data', df=df, index=False, format='csv')
```

Creating a project, setting the function into it, defining the URL with the data and running the function:

```python
source_url = mlrun.get_sample_path('data/batch-predict/training_set.parquet')
project = mlrun.get_or_create_project("data-items", "./", user_project=True)
data_prep_func = project.set_function("data-prep.py", name="data-prep", kind="job", image="mlrun/mlrun", handler="prep_data")
prep_data_run = data_prep_func.run(name='prep_data',
                                   handler=prep_data,
                                   inputs={'source_url': source_url},
                                   params={'label_column': 'label'})
```

To call the function with an `input` you can use the `inputs` dictionary attribute. To pass
a simple parameter, use the `params` dictionary attribute. The input value is the specific item uri
(per data store schema) as explained in [Shared data stores](../store/datastore.html#shared-data-stores).

From v1.3, `DataItem` objects are automatically parsed to the hinted type when a type hint is available.

Reading the data results from the run, you can easily get a run output artifact as a `DataItem` (so that you can view/use the artifact) using:

```python
# read the data locally as a Dataframe
prep_data_run.artifact('cleaned_data').as_df()
```

The {py:class}`~mlrun.datastore.DataItem` supports multiple convenience methods such as:
* **get()**, **put()** - to read/write data
* **download()**, **upload()** - to download/upload files
* **as_df()** - to convert the data to a DataFrame object
* **local** - to get a local file link to the data (that is downloaded locally if needed)
* **listdir()**, **stat** - file system like methods
* **meta** - access to the artifact metadata (in case of an artifact uri)
* **show()** - visualizes the data in Jupyter (as image, html, etc.)

See the **{py:class}`~mlrun.datastore.DataItem`** class documentation for details.

In order to get a DataItem object from a url use {py:func}`~mlrun.run.get_dataitem` or
{py:func}`~mlrun.run.get_object` (returns the `DataItem.get()`).

For example:

    df = mlrun.get_dataitem('s3://demo-data/mydata.csv').as_df()
    print(mlrun.get_object('https://my-site/data.json'))
