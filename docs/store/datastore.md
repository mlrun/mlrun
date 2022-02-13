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
* **s3** - S3 objects (AWS or other endpoints), format: `s3://<bucket>/path/to/file`
* **v3io, v3ios** - Iguazio v3io data fabric, format: `v3io://[<remote-host>]/<data-container>/path/to/file`
* **az** - Azure Blob Store, format: `az://<bucket>/path/to/file`
* **gs, gcs** - Google Cloud Storage objects, format: `gs://<bucket>/path/to/file`
* **store** - MLRun versioned artifacts [(see Artifacts)](./artifacts.md), format: `store://artifacts/<project>/<artifact-name>[:tag]`
* **memory** - in memory data registry for passing data within the same process, format `memory://key`, 
  use `mlrun.datastore.set_in_memory_item(key, value)` to register in memory data items (byte buffers or DataFrames).

### Storage credentials and parameters
Data stores may require connection credentials, those can be provided through environment variables 
or project/job context secrets. The exact credentials depend on the type of the data store, and are listed in
the following table. Each parameter specified can be provided as an environment variable, or as a project-secret that
has the same key as the name of the parameter.

MLRun jobs executed remotely run in independent pods, with their own environment. When setting an environment 
variable in the development environment (for example Jupyter), this has no effect on the executing pods. Therefore, 
before executing jobs that require access to storage credentials, these need to be provided by assigning environment 
variables to the MLRun runtime itself, assigning secrets to it, or placing the variables in project-secrets.

```{warning}
Passing secrets as environment variables to runtimes is discouraged, as they are exposed in the pod spec.
Refer to [Working with secrets](../secrets.md) for details on secret handling in MLRun.
```

For example, running a function locally:

```python
# Access object in AWS S3, in the "input-data" bucket 
source_url = "s3://input-data/input_data.csv"

os.environ["AWS_ACCESS_KEY_ID"] = "<access key ID>"
os.environ["AWS_SECRET_ACCESS_KEY"] = "<access key>"

# Execute a function that reads from the object pointed at by source_url.
# When running locally, the function can use the local environment variables.
local_run = func.run(name='aws_test', inputs={'source_url': source_url}, local=True)
```

Running the same function remotely:

```python
# Executing the function remotely using env variables (not recommended!)
func.set_env("AWS_ACCESS_KEY_ID", "<access key ID>").set_env("AWS_SECRET_ACCESS_KEY", "<access key>")
remote_run = func.run(name='aws_test', inputs={'source_url': source_url})

# Using project-secrets (recommended) - project secrets are automatically mounted to project functions
secrets = {"AWS_ACCESS_KEY_ID": "<access key ID>", "AWS_SECRET_ACCESS_KEY": "<access key>"}
db = mlrun.get_run_db()
db.create_project_secrets(project=project_name, provider="kubernetes", secrets=secrets)

remote_run = func.run(name='aws_test', inputs={'source_url': source_url})
```

The following sections list the credentials and configuration parameters applicable to each storage type.

#### v3io
When running in an Iguazio system, MLRun will automatically configure executed functions to use `v3io` storage, and will
pass the needed parameters (such as access-key) for authentication. Refer to the 
[auto-mount](Function_storage_auto_mount) section for more details on this process.

In some cases, v3io configuration needs to be overridden. The following parameters may be configured:

* `V3IO_API` - URL pointing to the v3io web-API service.
* `V3IO_ACCESS_KEY` - access key used to authenticate with the web API.
* `V3IO_USERNAME` - the user-name authenticating with v3io. While not strictly required when using an access-key to 
authenticate, it is used in several use-cases, such as resolving paths to home-directory.

#### S3
* `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - [access key](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html)
  parameters
* `S3_ENDPOINT_URL` - the S3 endpoint to use. If not specified, will default to AWS. For example, to access 
  a storage bucket in Wasabi storage, use `S3_ENDPOINT_URL = "https://s3.wasabisys.com"`

#### Azure blob storage
Azure blob storage can utilize several methods of authentication, each require a different set of parameters as listed
here:

| Authentication method | Parameters |
|-----------------------|------------|
| [Connection string](https://docs.microsoft.com/en-us/azure/storage/common/storage-configure-connection-string) | `AZURE_STORAGE_CONNECTION_STRING` |
| [SAS token](https://docs.microsoft.com/en-us/azure/storage/common/storage-sas-overview#sas-token) | `AZURE_STORAGE_ACCOUNT_NAME`<br/>`AZURE_STORAGE_SAS_TOKEN` |
| [Account key](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?tabs=azure-portal) | `AZURE_STORAGE_ACCOUNT_NAME`<br/>`AZURE_STORAGE_ACCOUNT_KEY` |
| [Service principal with a client secret](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal) | `AZURE_STORAGE_ACCOUNT_NAME`<br/>`AZURE_STORAGE_CLIENT_ID`<br/>`AZURE_STORAGE_CLIENT_SECRET`<br/>`AZURE_STORAGE_TENANT_ID` |

#### Google cloud storage
* `GOOGLE_APPLICATION_CREDENTIALS` - path to the application credentials to use (in the form of a JSON file). This can
be used if this file is located in a location on shared storage, accessible to pods executing MLRun jobs.
* `GCP_CREDENTIALS` - when the credentials file cannot be mounted to the pod, this environment variable may contain
the contents of this file. If configured in the function pod, MLRun will dump its contents to a temporary file 
and point `GOOGLE_APPLICATION_CREDENTIALS` at it.

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

Reading the data results from our run:
we can easily get a run output artifact as a `DataItem` (allowing us to view/use the artifact) using:

```python
# read the data locally as a Dataframe
prep_data_run.artifact('cleaned_data').as_df()
```

The {py:class}`~mlrun.datastore.DataItem` support multiple convenience methods such as:
* **get()**, **put()** - to read/write data
* **download()**, **upload()** - to download/upload files
* **as_df()** - to convert the data to a DataFrame object
* **local** - to get a local file link to the data (will be downloaded locally if needed)
* **listdir()**, **stat** - file system like methods
* **meta** - access to the artifact metadata (in case of an artifact uri)
* **show()** - will visualize the data in Jupyter (as image, html, etc.)

Check the **{py:class}`~mlrun.datastore.DataItem`** class documentation for details

In order to get a DataItem object from a url use {py:func}`~mlrun.run.get_dataitem` or 
{py:func}`~mlrun.run.get_object` (returns the `DataItem.get()`), for example:

    df = mlrun.get_dataitem('s3://demo-data/mydata.csv').as_df()
    print(mlrun.get_object('https://my-site/data.json'))

