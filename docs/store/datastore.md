(datastore)=
# Data stores

A data store defines a storage provider (e.g. file system, S3, Azure blob, Iguazio v3io, etc.).

MLRun supports multiple data stores. Additional data stores, for example MongoDB, can easily be added by extending the `DataStore` class.

Data stores are referred to using the schema prefix (e.g. `s3://my-bucket/path`). The currently supported schemas and their urls:
* **files** &mdash; local/shared file paths, format: `/file-dir/path/to/file` (Unix) or `C:/dir/file` (Windows)
* **http, https** &mdash; read data from HTTP sources (read-only), format: `https://host/path/to/file` (Not supported by runtimes: Spark and RemoteSpark)
* **az** &mdash; Azure Blob storage, format: `az://<container>/path/to/file`
* **dbfs** &mdash; Databricks storage, format: `dbfs://path/to/file` (Not supported by runtimes spark and remote-spark)
* **gs, gcs** &mdash; Google Cloud Storage objects, format: `gs://<bucket>/path/to/file`
* **s3** &mdash; S3 objects (AWS or other endpoints), format: `s3://<bucket>/path/to/file`
* **v3io, v3ios** &mdash; Iguazio v3io data fabric, format: `v3io://<data-container>/path/to/file`
* **store** &mdash; MLRun versioned artifacts [(see Artifacts)](./artifacts.html), format: `store://artifacts/<project>/<artifact-name>[:tag]`
* **memory** &mdash; in memory data registry for passing data within the same process, format `memory://key`, use `mlrun.datastore.set_in_memory_item(key, value)` 
   to register in memory data items (byte buffers or DataFrames). (Not supported by all Spark runtimes)

**In this section**
- [Storage credentials and parameters](#storage-credentials-and-parameters)
- [Data store profiles](#data-store-profiles)
- [Alibaba Cloud Object Storage Service (OSS)](#alibaba-cloud-object-storage-service-oss)
- [Azure data store](#azure-data-store)
- [Databricks file system](#databricks-file-system)
- [Google cloud storage](#google-cloud-storage)
- [HDFS](#hdfs)
- [S3](#s3)
- [V3IO](#v3io)


## Storage credentials and parameters
Data stores might require connection credentials. These can be provided through environment variables 
or project/job context secrets. The exact credentials depend on the type of the data store. They are 
listed in the following sections. Each parameter specified can be provided as an environment variable, 
or as a project-secret that has the same key as the name of the parameter.

MLRun jobs that are executed remotely run in independent pods, with their own environment. When setting an environment 
variable in the development environment (for example Jupyter), this has no effect on the executing pods. 
Therefore, before executing jobs that require access to storage credentials, these need to be provided 
by assigning environment variables to the MLRun runtime itself, assigning secrets to it, or placing 
the variables in project-secrets.

You can also use [data store profiles](#data-store-profiles) to provide credentials.

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
local_run = func.run(name="aws_func", inputs={"source_url": source_url}, local=True)
```

Running the same function remotely:

```python
# Executing the function remotely using env variables (not recommended!)
func.set_env("AWS_ACCESS_KEY_ID", "<access key ID>").set_env(
    "AWS_SECRET_ACCESS_KEY", "<access key>"
)
remote_run = func.run(name="aws_func", inputs={"source_url": source_url})

# Using project-secrets (recommended) - project secrets are automatically mounted to project functions
secrets = {
    "AWS_ACCESS_KEY_ID": "<access key ID>",
    "AWS_SECRET_ACCESS_KEY": "<access key>",
}
db = mlrun.get_run_db()
db.create_project_secrets(project=project_name, provider="kubernetes", secrets=secrets)

remote_run = func.run(name="aws_func", inputs={"source_url": source_url})
```
  
## Data store profiles

```{admonition} Note
Datastore profiles are not part of a project export/import.
```

You can use a data store profile to manage datastore credentials. A data store profile 
holds all the information required to address an external data source, including credentials. 
You can create 
multiple profiles for one datasource. For example, 
two different Redis data stores with different credentials. Targets, sources, and artifacts, 
can all use the data store profile by using the `ds://<profile-name>` convention.
After you create a profile object, you make it available on remote pods by calling 
`project.register_datastore_profile`.

Create a data store profile in the context of a project. Example of creating a Redis datastore profile:
1. Create the profile, for example:<br>
   `profile = DatastoreProfileRedis(name="profile-name", endpoint_url="redis://11.22.33.44:6379", username="user", password="password")`
    The username and password parameters are optional. 
2. Register it within the project:<br>
   `project.register_datastore_profile(profile)`
2. Use the profile by specifying the 'ds' URI scheme. For example:<br>
   `RedisNoSqlTarget(path="ds://profile-name/a/b")`
   
More options:
- To access a profile from the client/sdk, register the profile locally by calling
   `register_temporary_client_datastore_profile()` with a profile object.
- You can also choose to retrieve the public information of an already registered profile by calling 
   `project.get_datastore_profile()` and then adding the private credentials before registering it locally.

    For example, using Redis:
    ```python
    redis_profile = project.get_datastore_profile("my_profile")
    local_redis_profile = DatastoreProfileRedis(
        redis_profile.name,
        redis_profile.endpoint_url,
        username="mylocaluser",
        password="mylocalpassword",
    )
    register_temporary_client_datastore_profile(local_redis_profile)
    ```

```{admonition} Note
Data store profiles do not support: v3io (datastore, or source/target), snowflake source, DBFS for spark runtimes, Dask runtime.
```

See also:
- {py:class}`~mlrun.projects.MlrunProject.list_datastore_profiles` 
- {py:class}`~mlrun.projects.MlrunProject.get_datastore_profile`
- {py:class}`~mlrun.datastore.datastore_profile.register_temporary_client_datastore_profile`
- {py:class}`~mlrun.projects.MlrunProject.delete_datastore_profile`

The methods `get_datastore_profile()` and `list_datastore_profiles()` only return public information about 
the profiles. Access to private attributes is restricted to applications running in Kubernetes pods.

## Alibaba Cloud Object Storage Service (OSS)

### Alibaba Cloud Object Storage Service credentials and parameters

* `ALIBABA_ACCESS_KEY_ID`, `ALIBABA_SECRET_ACCESS_KEY` &mdash; [access key](https://www.alibabacloud.com/help/en/oss/developer-reference/authorize-access-3)
  parameters
* `ALIBABA_ENDPOINT_URL` &mdash; The OSS endpoint to use, for example: https://oss-cn-hangzhou.aliyuncs.com


## Azure data store

### Azure Blob storage credentials and parameters

The Azure Blob storage can utilize several methods of authentication. Each requires a different set of parameters as listed
here:

| Authentication method | Parameters |
|-----------------------|------------|
| [Connection string](https://docs.microsoft.com/en-us/azure/storage/common/storage-configure-connection-string) | `AZURE_STORAGE_CONNECTION_STRING` |
| [SAS token](https://docs.microsoft.com/en-us/azure/storage/common/storage-sas-overview#sas-token) | `AZURE_STORAGE_ACCOUNT_NAME`<br/>`AZURE_STORAGE_SAS_TOKEN` |
| [Account key](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?tabs=azure-portal) | `AZURE_STORAGE_ACCOUNT_NAME`<br/>`AZURE_STORAGE_ACCOUNT_KEY` |
| [Service principal with a client secret](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal) | `AZURE_STORAGE_ACCOUNT_NAME`<br/>`AZURE_STORAGE_CLIENT_ID`<br/>`AZURE_STORAGE_CLIENT_SECRET`<br/>`AZURE_STORAGE_TENANT_ID` |

```{note}
The `AZURE_STORAGE_CONNECTION_STRING` configuration uses the `BlobServiceClient` to access objects. This has
limited functionality and cannot be used to access Azure Data Lake storage objects. In this case use one of the other 
authentication methods that use the `fsspec` mechanism. 
```

### Azure data store profile
```python
profile = DatastoreProfileAzureBlob(
    name="profile-name", connection_string=connection_string
)
ParquetTarget(path="ds://profile-name/az_blob/path/to/parquet.pq")
```

`DatastoreProfileAzureBlob` init parameters:
- `name` &mdash; Name of the profile.
- `connection_string` &mdash; The Azure connection string that points at a storage account.
For privacy reasons, it's tagged as a private attribute, and its default value is `None`.
The equivalent to this parameter in environment authentication is "AZURE_STORAGE_CONNECTION_STRING".
for example:<br>
`DefaultEndpointsProtocol=https;AccountName=myAcct;AccountKey=XXXX;EndpointSuffix=core.windows.net`

The following variables allow alternative methods of authentication. All of these variables require 
`account_name`.
- `account_name` &mdash; This parameter represents the name of the Azure Storage account.
Each Azure Storage account has a unique name, and it serves as a globally-unique identifier for the storage account within the Azure cloud.
The equivalent to this parameter in environment authentication is "AZURE_STORAGE_ACCOUNT_NAME".
- `account_key` &mdash; The storage account key is a security credential associated with an Azure Storage account.
It is a primary access key used for authentication and authorization purposes.
This key is sensitive information and is kept confidential.
The equivalent to this parameter in environment authentication is "AZURE_STORAGE_ACCOUNT_KEY".
- `sas_token` &mdash; Shared Access Signature (SAS) token for time-bound access.
This token is sensitive information. Equivalent to "AZURE_STORAGE_SAS_TOKEN" in environment authentication.

Authentication against Azure services using a Service Principal:
- `client_id` &mdash; This variable holds the client ID associated with an Azure Active Directory (AAD) application,
which represents the Service Principal. In Azure, a Service Principal is used for non-interactive authentication, allowing applications to access Azure resources.
The equivalent to this parameter in environment authentication is "AZURE_STORAGE_CLIENT_ID".
- `client_secret` &mdash; This variable stores the client secret associated with the Azure AD application (Service Principal).
The client secret is a credential that proves the identity of the application when it requests access to Azure resources.
This key is sensitive information and is kept confidential.
The equivalent to this parameter in environment authentication is "AZURE_STORAGE_CLIENT_SECRET".
- `tenant_id` &mdash; This variable holds the Azure AD tenant ID, which uniquely identifies the organization or directory in Azure Active Directory.
The equivalent to this parameter in environment authentication is "AZURE_STORAGE_TENANT_ID".

Credential authentication:
- `credential` &mdash; TokenCredential or SAS token. The credentials with which to authenticate.
This variable is sensitive information and is kept confidential.
- `container` &mdash; A string representing the container. When specified, it is automatically prepended to the object path, and thus, it should not be manually included in the target path by the user.
This parameter will become mandatory starting with version 1.9.

## Databricks file system 
### DBFS credentials and parameters

```{Admonition} Note
Not supported by the spark and remote-spark runtimes.
```
* `DATABRICKS_HOST` &mdash; hostname in the format: https://abc-d1e2345f-a6b2.cloud.databricks.com'
* `DATABRICKS_TOKEN` &mdash; Databricks access token. 
   Perform [Databricks personal access token authentication](https://docs.databricks.com/en/dev-tools/auth.html#databricks-personal-access-token-authentication).
   
### DBFS data store profile

```python
profile = DatastoreProfileDBFS(
    name="profile-name",
    endpoint_url="abc-d1e2345f-a6b2.cloud.databricks.com",
    token=token,
)
ParquetTarget(path="ds://profile-name/path/to/parquet.pq")
```

`DatastoreProfileDBFS` init parameters:
- `name` &mdash; Name of the profile.
- `endpoint_url` &mdash; A string representing the endpoint URL of the DBFS service.
The equivalent to this parameter in environment authentication is "DATABRICKS_HOST".
- `token` &mdash; A string representing the secret key used for authentication to the DBFS service. 
For privacy reasons, it's tagged as a private attribute, and its default value is `None`.
The equivalent to this parameter in environment authentication is "DATABRICKS_TOKEN".

## Google cloud storage

### GCS credentials and parameters
* `GOOGLE_APPLICATION_CREDENTIALS` &mdash; Path to the application credentials to use (in the form of a JSON file). This can
be used if this file is located in a location on shared storage, accessible to pods executing MLRun jobs.
* `GCP_CREDENTIALS` &mdash; When the credentials file cannot be mounted to the pod, this secret or environment variable
may contain the contents of this file. If configured in the function pod, MLRun dumps its contents to a temporary file 
and points `GOOGLE_APPLICATION_CREDENTIALS` at it. An exception is `BigQuerySource`, which passes `GCP_CREDENTIALS`'s
contents directly to the query engine.

### GCS data store profile

```python
profile = DatastoreProfileGCS(
    name="profile-name", credentials_path="/local_path/to/gcs_credentials.json"
)
ParquetTarget(path="ds://profile-name/gcs_bucket/path/to/parquet.pq")
```

`DatastoreProfileGCS` init parameters:
- `name` &mdash; Name of the profile.
- `credentials_path` &mdash; A string representing the local JSON file path that contains the authentication parameters required by the GCS API.
The equivalent to this parameter in environment authentication is "GOOGLE_APPLICATION_CREDENTIALS."
- `gcp_credentials` &mdash; A JSON in a string format representing the authentication parameters required by GCS API. 
For privacy reasons, it's tagged as a private attribute, and its default value is `None`.
The equivalent to this parameter in environment authentication is "GCP_CREDENTIALS".
- `bucket` &mdash; A string representing the bucket. When specified, it is automatically prepended to the object path, and thus, it should not be manually included in the target path by the user. 
This parameter will become mandatory starting with version 1.9.

The code prioritizes `gcp_credentials` over `credentials_path`.



## HDFS

MLRun supports HDFS only with datastore profiles, and not with env-vars.


### HDFS data store profile

```python
profile = DatastoreProfileHdfs(name="profile-name")
ParquetTarget(path="ds://profile-name/path/to/parquet.pq")
```

`DatastoreProfileHdfs` init parameters:
- `name` &mdash; Name of the profile
- `host` &mdash; HDFS namenode host
- `port` &mdash; HDFS namenode port
- `http_port` &mdash; WebHDFS port
- `user` &mdash; User name. Only affects WebHDFS. When using Spark, or when this parameter is not defined, the user name is the value of the `HADOOP_USER_NAME` environment variable. If this environment variable is also not defined, the current user's user name is used. In Spark, this is evaluated at the time that the spark context is created.

You can set `HADOOP_USER_NAME` locally as follows: 
```python
import os

os.environ["HADOOP_USER_NAME"] = "..."
```

An example of registering an HDFS data store profile and using it as described in [Data store profiles](#data-store-profiles):
```python
DatastoreProfileHdfs(
    name="my-hdfs",
    host="localhost",
    port=9000,
    http_port=9870,
)
```

To set it on a function, use:
```python
function.spec.env.append({"name": "HADOOP_USER_NAME", "value": "galt"})
```

In feature store, you can set it via `RunConfig`:
```python
from mlrun.feature_store.common import RunConfig

run_config = RunConfig(
    local=False,
    kind="remote-spark",
    extra_spec={"spec": {"env": [{"name": "HADOOP_USER_NAME", "value": "galt"}]}},
)
feature_set.ingest(..., run_config=run_config)
```

## S3

### S3 credentials and parameters

* `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` &mdash; [access key](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html)
  parameters
* `S3_ENDPOINT_URL` &mdash; The S3 endpoint to use. If not specified, it defaults to AWS. For example, to access 
  a storage bucket in Wasabi storage, use `S3_ENDPOINT_URL = "https://s3.wasabisys.com"`
* `MLRUN_AWS_ROLE_ARN` &mdash; [IAM role to assume](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-api.html). 
  Connect to AWS using the secret key and access key, and assume the role whose ARN is provided. The 
  ARN must be of the format `arn:aws:iam::<account-of-role-to-assume>:role/<name-of-role>`
* `AWS_PROFILE` &mdash; Name of credentials profile from a local AWS credentials file. 
  When using a profile, the authentication secrets (if defined) are ignored, and credentials are retrieved from the 
  file. This option should be used for local development where AWS credentials already exist (created by `aws` CLI, for
  example)
  
### S3 data store profile

```python
profile = DatastoreProfileS3(name="profile-name")
ParquetTarget(path="ds://profile-name/aws_bucket/path/to/parquet.pq")
```

`DatastoreProfileS3` init parameters:
- `name` &mdash; Name of the profile
- `endpoint_url` &mdash; A string representing the endpoint URL of the S3 service. It's typically required for non-AWS S3-compatible services. If not provided, the default is `None`. The equivalent to this parameter in environment authentication is env["S3_ENDPOINT_URL"].
- `force_non_anonymous` &mdash; A string that determines whether to force non-anonymous access to the S3 bucket. The default value is `None`, meaning the behavior is not explicitly set. The equivalent to this parameter in environment authentication is env["S3_NON_ANONYMOUS"].
- `profile_name` &mdash; A string representing the name of the profile. This might be used to refer to specific named configurations for connecting to S3. The default value is `None`. The equivalent to this parameter in environment authentication is env["AWS_PROFILE"].
- `assume_role_arn` &mdash; A string representing the Amazon Resource Name (ARN) of the role to assume when interacting with the S3 service. This can be useful for granting temporary permissions. By default, it is set to `None`. The equivalent to this parameter in environment authentication is env["MLRUN_AWS_ROLE_ARN"]
- `access_key_id` &mdash; A string representing the access key used for authentication to the S3 service. It's one of the credentials parts when you're not using anonymous access or IAM roles. For privacy reasons, it's tagged as a private attribute, and its default value is `None`. The equivalent to this parameter in environment authentication is env["AWS_ACCESS_KEY_ID"].
- `secret_key` &mdash; A string representing the secret key, which pairs with the access key, used for authentication to the S3 service. It's the second part of the credentials when not using anonymous access or IAM roles. It's also tagged as private for privacy and security reasons. The default value is `None`. The equivalent to this parameter in environment authentication is env["AWS_SECRET_ACCESS_KEY"].
- `bucket` &mdash; A string representing the bucket. When specified, it is automatically prepended to the object path, and thus, it should not be manually included in the target path by the user. 
This parameter will become mandatory starting with version 1.9.


## V3IO 

### V3IO credentials and parameters
When running in an Iguazio system, MLRun automatically configures the executed functions to use `v3io` storage, and passes 
the needed parameters (such as access-key) for authentication. Refer to the 
[auto-mount](../runtimes/function-storage.html) section for more details on this process.

In some cases, the V3IO configuration needs to be overridden. The following parameters can be configured:

* `V3IO_API` &mdash; URL pointing to the V3IO web-API service.
* `V3IO_ACCESS_KEY` &mdash; Access key used to authenticate with the web API.
* `V3IO_USERNAME` &mdash; The user-name authenticating with V3IO. While not strictly required when using an access-key to 
authenticate, it is used in several use-cases, such as resolving paths to the home-directory.

### V3IO data store profile
```python
profile = DatastoreProfileV3io(
    name="test_profile", v3io_access_key="12345678-1234-1234-1234-123456789012"
)
register_temporary_client_datastore_profile(
    profile
) or project.register_datastore_profile(profile)
ParquetTarget(path="ds://test_profile/container/path/to/parquet.pq")
```

`DatastoreProfileV3io` init parameters:
- `name` &mdash; Name of the profile
- `v3io_access_key` &mdash; Optional. Access key to the remote Iguazio cluster. If not provided, the default is value is taken from the environment variable "V3IO_ACCESS_KEY". For privacy reasons, it's tagged as a private attribute.


% ## Adding a data store profile Return to doc when there are personas

% If you already have a functioning datastore, integrating it with a datastore profile is straightforward. Follow these steps:
% 1. Derive a new datastore profile class from the `DatastoreProfile` class. During this process, specify the datastore profile type. 
%    This is usually the same as the schema associated with the datastore URL, although this is not strictly necessary.
% 2. Incorporate all necessary parameters for accessing the datastore, ensuring they are appropriately classified as public or private.
% 3. Implement two essential methods: `secrets()` and `url()`.
%    - The `url()` method constructs a URL path to a specific object. It takes a 'subpath' parameter, which is the relative path to the object, 
%    and returns the fully resolved URL that is used to access the object in the datastore.
%    - The `secrets()` method returns a dictionary containing environment variables that are used when accessing the datastore.
% 4. In the `create_from_json()` function, introduce factory settings for the newly created profile. Use the profile type as a key for these settings.

## See also
- {py:class}`~mlrun.projects.MlrunProject.list_datastore_profiles` 
- {py:class}`~mlrun.projects.MlrunProject.get_datastore_profile`
- {py:class}`~mlrun.datastore.datastore_profile.register_temporary_client_datastore_profile`
- {py:class}`~mlrun.projects.MlrunProject.delete_datastore_profile`

The methods `get_datastore_profile()` and `list_datastore_profiles()` only return public information about 
the profiles. Access to private attributes is restricted to applications running in Kubernetes pods.
