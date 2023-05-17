(datastore)=
# Data stores

A data store defines a storage provider (e.g. file system, S3, Azure blob, Iguazio v3io, etc.).

**In this section**
- [Shared data stores](#shared-data-stores)
- [Storage credentials and parameters](#storage-credentials-and-parameters)
   
## Shared data stores

MLRun supports multiple data stores. (More can easily added by extending the `DataStore` class.)
Data stores are referred to using the schema prefix (e.g. `s3://my-bucket/path`). The currently supported schemas and their urls:
* **files** &mdash; local/shared file paths, format: `/file-dir/path/to/file` (Unix) or `C:/dir/file` (Windows)
* **http, https** &mdash; read data from HTTP sources (read-only), format: `https://host/path/to/file`
* **s3** &mdash; S3 objects (AWS or other endpoints), format: `s3://<bucket>/path/to/file`
* **v3io, v3ios** &mdash; Iguazio v3io data fabric, format: `v3io://[<remote-host>]/<data-container>/path/to/file`
* **az** &mdash; Azure Blob storage, format: `az://<container>/path/to/file`
* **gs, gcs** &mdash; Google Cloud Storage objects, format: `gs://<bucket>/path/to/file`
* **store** &mdash; MLRun versioned artifacts [(see Artifacts)](./artifacts.html), format: `store://artifacts/<project>/<artifact-name>[:tag]`
* **memory** &mdash; in memory data registry for passing data within the same process, format `memory://key`, use `mlrun.datastore.set_in_memory_item(key, value)` to register in memory data items (byte buffers or DataFrames).

## Storage credentials and parameters
Data stores might require connection credentials. These can be provided through environment variables 
or project/job context secrets. The exact credentials depend on the type of the data store and are listed in
the following table. Each parameter specified can be provided as an environment variable, or as a project-secret that
has the same key as the name of the parameter.

MLRun jobs executed remotely run in independent pods, with their own environment. When setting an environment 
variable in the development environment (for example Jupyter), this has no effect on the executing pods. Therefore, 
before executing jobs that require access to storage credentials, these need to be provided by assigning environment 
variables to the MLRun runtime itself, assigning secrets to it, or placing the variables in project-secrets.

```{warning}
Passing secrets as environment variables to runtimes is discouraged, as they are exposed in the pod spec.
Refer to [Working with secrets](../secrets.html) for details on secret handling in MLRun.
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

### v3io
When running in an Iguazio system, MLRun automatically configures executed functions to use `v3io` storage, and passes 
the needed parameters (such as access-key) for authentication. Refer to the 
[auto-mount](../runtimes/function-storage.html) section for more details on this process.

In some cases, the v3io configuration needs to be overridden. The following parameters can be configured:

* `V3IO_API` &mdash; URL pointing to the v3io web-API service.
* `V3IO_ACCESS_KEY` &mdash; access key used to authenticate with the web API.
* `V3IO_USERNAME` &mdash; the user-name authenticating with v3io. While not strictly required when using an access-key to 
authenticate, it is used in several use-cases, such as resolving paths to the home-directory.

### S3
* `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` &mdash; [access key](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html)
  parameters
* `S3_ENDPOINT_URL` &mdash; the S3 endpoint to use. If not specified, it defaults to AWS. For example, to access 
  a storage bucket in Wasabi storage, use `S3_ENDPOINT_URL = "https://s3.wasabisys.com"`
* `MLRUN_AWS_ROLE_ARN` &mdash; [IAM role to assume](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-api.html). 
  Connect to AWS using the secret key and access key, and assume the role whose ARN is provided. The 
  ARN must be of the format `arn:aws:iam::<account-of-role-to-assume>:role/<name-of-role>`
* `AWS_PROFILE` &mdash; name of credentials profile from a local AWS credentials file. 
  When using a profile, the authentication secrets (if defined) are ignored, and credentials are retrieved from the 
  file. This option should be used for local development where AWS credentials already exist (created by `aws` CLI, for
  example)

### Azure Blob storage
The Azure Blob storage can utilize several methods of authentication. Each requires a different set of parameters as listed
here:

| Authentication method | Parameters |
|-----------------------|------------|
| [Connection string](https://docs.microsoft.com/en-us/azure/storage/common/storage-configure-connection-string) | `AZURE_STORAGE_CONNECTION_STRING` |
| [SAS token](https://docs.microsoft.com/en-us/azure/storage/common/storage-sas-overview#sas-token) | `AZURE_STORAGE_ACCOUNT_NAME`<br/>`AZURE_STORAGE_SAS_TOKEN` |
| [Account key](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?tabs=azure-portal) | `AZURE_STORAGE_ACCOUNT_NAME`<br/>`AZURE_STORAGE_KEY` |
| [Service principal with a client secret](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal) | `AZURE_STORAGE_ACCOUNT_NAME`<br/>`AZURE_STORAGE_CLIENT_ID`<br/>`AZURE_STORAGE_CLIENT_SECRET`<br/>`AZURE_STORAGE_TENANT_ID` |

```{note}
The `AZURE_STORAGE_CONNECTION_STRING` configuration uses the `BlobServiceClient` to access objects. This has
limited functionality and cannot be used to access Azure Datalake storage objects. In this case use one of the other 
authentication methods that use the `fsspec` mechanism. 
```

### Google cloud storage
* `GOOGLE_APPLICATION_CREDENTIALS` &mdash; path to the application credentials to use (in the form of a JSON file). This can
be used if this file is located in a location on shared storage, accessible to pods executing MLRun jobs.
* `GCP_CREDENTIALS` &mdash; when the credentials file cannot be mounted to the pod, this secret or environment variable
may contain the contents of this file. If configured in the function pod, MLRun dumps its contents to a temporary file 
and points `GOOGLE_APPLICATION_CREDENTIALS` at it. An exception is `BigQuerySource`, which passes `GCP_CREDENTIALS`'s
contents directly to the query engine.