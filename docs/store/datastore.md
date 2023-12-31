(datastore)=
# Data stores

A data store defines a storage provider (e.g. file system, S3, Azure blob, Iguazio v3io, etc.).

**In this section**
- [Shared data stores](#shared-data-stores)
- [Storage credentials and parameters](#storage-credentials-and-parameters)
- [Using data store profiles](#using-data-store-profiles)
   
## Shared data stores

MLRun supports multiple data stores. (More can easily added by extending the `DataStore` class.)
Data stores are referred to using the schema prefix (e.g. `s3://my-bucket/path`). The currently supported schemas and their urls:
* **files** &mdash; local/shared file paths, format: `/file-dir/path/to/file` (Unix) or `C:/dir/file` (Windows)
* **http, https** &mdash; read data from HTTP sources (read-only), format: `https://host/path/to/file` (Not supported by runtimes spark and remote-spark)
* **s3** &mdash; S3 objects (AWS or other endpoints), format: `s3://<bucket>/path/to/file`
* **v3io, v3ios** &mdash; Iguazio v3io data fabric, format: `v3io://[<remote-host>]/<data-container>/path/to/file`
* **az** &mdash; Azure Blob storage, format: `az://<container>/path/to/file`
* **dbfs** &mdash; Databricks storage, format: `dbfs://path/to/file` (Not supported by runtimes spark and remote-spark)
* **gs, gcs** &mdash; Google Cloud Storage objects, format: `gs://<bucket>/path/to/file`
* **store** &mdash; MLRun versioned artifacts [(see Artifacts)](./artifacts.html), format: `store://artifacts/<project>/<artifact-name>[:tag]`
* **memory** &mdash; in memory data registry for passing data within the same process, format `memory://key`, use `mlrun.datastore.set_in_memory_item(key, value)` 
   to register in memory data items (byte buffers or DataFrames). (Not supported by all Spark runtimes)

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

You can also use [data store profiles](#using-data-store-profiles) to provide credentials.

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
When running in an Iguazio system, MLRun automatically configures the executed functions to use `v3io` storage, and passes 
the needed parameters (such as access-key) for authentication. Refer to the 
[auto-mount](../runtimes/function-storage.html) section for more details on this process.

In some cases, the v3io configuration needs to be overridden. The following parameters can be configured:

* `V3IO_API` &mdash; URL pointing to the v3io web-API service.
* `V3IO_ACCESS_KEY` &mdash; access key used to authenticate with the web API.
* `V3IO_USERNAME` &mdash; the user-name authenticating with v3io. While not strictly required when using an access-key to 
authenticate, it is used in several use-cases, such as resolving paths to the home-directory.



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

### Databricks file system
```{Admonition} Note
Not supported by the spark and remote-spark runtimes.
```
* `DATABRICKS_HOST` &mdash; hostname in the format: https://abc-d1e2345f-a6b2.cloud.databricks.com'
* `DATABRICKS_TOKEN` &mdash; Databricks access token. 
   Perform [Databricks personal access token authentication](https://docs.databricks.com/en/dev-tools/auth.html#databricks-personal-access-token-authentication).

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
  
## Using data store profiles

You can use a data store profile to manage datastore credentials. A data store profile 
holds all the information required to address an external data source, including credentials. 
You can create 
multiple profiles for one datasource, for example, 
two different Redis data stores with different credentials. Targets, sources, and artifacts, 
can all use the data store profile by using the `ds://<profile-name>` convention.
After you create a profile object, you make it available on remote pods by calling 
`project.register_datastore_profile`.

Create a data store profile in the context of a project. Example of creating a Redis datastore profile:
1. Create the profile, for example:<br>
   `profile = DatastoreProfileRedis(name="test_profile", endpoint_url="redis://11.22.33.44:6379", username="user", password="password")`
    The username and password parameters are optional. 
2. Register it within the project:<br>
   `project.register_datastore_profile(profile)`
2. Use the profile by specifying the 'ds' URI scheme. For example:<br>
   `RedisNoSqlTarget(path="ds://test_profile/a/b")`<br>
    If you want to use a profile from a different project, you can specify it 
	explicitly in the URI using the format:<br>
    `RedisNoSqlTarget(path="ds://another_project@test_profile")`


To access a profile from the client/sdk, register the profile locally by calling
   `register_temporary_client_datastore_profile()` with a profile object.
You can also choose to retrieve the public information of an already registered profile by calling 
   `project.get_datastore_profile()` and then adding the private credentials before registering it locally.
For example, using Redis:
```
redis_profile = project.get_datastore_profile("my_profile")
local_redis_profile = DatastoreProfileRedis(redis_profile.name, redis_profile.endpoint_url, username="mylocaluser", password="mylocalpassword")
register_temporary_client_datastore_profile(local_redis_profile)
```
```{admonition} Note
Datastore profile does not support: v3io (datastore, or source/target), snowflake source, DBFS for spark runtimes, Dask runtime.
```


### Azure data store profile
```
profile = DatastoreProfileAzureBlob(name="test_profile",connection_string=connection_string)
ParquetTarget(path="ds://test_profile/az_blob/path/to/parquet.pq")
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
This token is ensitive information. Equivalent to "AZURE_STORAGE_SAS_TOKEN" in environment authentication.

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
credential authentication:
- `credential` &mdash; TokenCredential or SAS token. The credentials with which to authenticate.
This variable is sensitive information and is kept confidential.

### DBFS data store profile

```
profile = DatastoreProfileDBFS(name="test_profile", endpoint_url="abc-d1e2345f-a6b2.cloud.databricks.com", token=token)
ParquetTarget(path="ds://test_profile/path/to/parquet.pq")
```

`DatastoreProfileDBFS` init parameters:
- `name` &mdash; Name of the profile.
- `endpoint_url` &mdash; A string representing the endpoint URL of the DBFS service.
The equivalent to this parameter in environment authentication is "DATABRICKS_HOST".
- `token` &mdash; A string representing the secret key used for authentication to the DBFS service. 
For privacy reasons, it's tagged as a private attribute, and its default value is `None`.
The equivalent to this parameter in environment authentication is "DATABRICKS_TOKEN".

### GCS data store profile
```
profile = DatastoreProfileGCS(name="test_profile",credentials_path="/local_path/to/gcs_credentials.json")
ParquetTarget(path="ds://test_profile/gcs_bucket/path/to/parquet.pq")
```

`DatastoreProfileGCS` init parameters:
- `name` &mdash; Name of the profile.
- `credentials_path` &mdash; A string representing the local JSON file path that contains the authentication parameters required by the GCS API.
The equivalent to this parameter in environment authentication is "GOOGLE_APPLICATION_CREDENTIALS."
- `gcp_credentials` &mdash; A JSON in a string format representing the authentication parameters required by GCS API. 
For privacy reasons, it's tagged as a private attribute, and its default value is `None`.
The equivalent to this parameter in environment authentication is "GCP_CREDENTIALS".
The code prioritizes `credentials_path` over `gcp_credentials` if both are not None.


### Kafka data store profile

```
profile = DatastoreProfileKafkaTarget(name="test_profile",bootstrap_servers="localhost", topic="test_topic")
target = KafkaTarget(path="ds://test_profile")
```

`DatastoreProfileKafkaTarget` class parameters:
- `name` &mdash; Name of the profile
- `bootstrap_servers` &mdash; A string representing the 'bootstrap servers' for Kafka. These are the initial contact points you use to discover the full set of servers in the Kafka cluster, typically provided in the format `host1:port1,host2:port2,...`.
- `topic` &mdash; A string that denotes the Kafka topic to which data will be sent or from which data will be received.
- `kwargs_public` &mdash; This is a dictionary (`Dict`) meant to hold a collection of key-value pairs that could represent settings or configurations deemed public. These pairs are subsequently passed as parameters to the underlying `kafka.KafkaConsumer()` constructor. The default value for `kwargs_public` is `None`.
- `kwargs_private` &mdash; This dictionary (`Dict`) is designed to store key-value pairs, typically representing configurations that are of a private or sensitive nature. These pairs are also passed as parameters to the underlying `kafka.KafkaConsumer()` constructor. It defaults to `None`.


```
profile = DatastoreProfileKafkaSource(name="test_profile",bootstrap_servers="localhost", topic="test_topic")
target = KafkaSource(path="ds://test_profile")
```

`DatastoreProfileKafkaSource` class parameters:
- `name` &mdash; Name of the profile
- `brokers` &mdash; This parameter can either be a single string or a list of strings representing the Kafka brokers. Brokers serve as the contact points for clients to connect to the Kafka cluster.
- `topics` &mdash; A string or list of strings that denote the Kafka topics from which data will be sourced or read.
- `group` &mdash; A string representing the consumer group name. Consumer groups are used in Kafka to allow multiple consumers to coordinate and consume messages from topics. The default consumer group is set to `"serving"`.
- `initial_offset` &mdash; A string that defines the starting point for the Kafka consumer. It can be set to `"earliest"` to start consuming from the beginning of the topic, or `"latest"` to start consuming new messages only. The default is `"earliest"`.
- `partitions` &mdash; This can either be a single string or a list of strings representing the specific partitions from which the consumer should read. If not specified, the consumer can read from all partitions.
- `sasl_user` &mdash; A string representing the username for SASL authentication, if required by the Kafka cluster. It's tagged as private for security reasons.
- `sasl_pass` &mdash; A string representing the password for SASL authentication, correlating with the `sasl_user`. It's tagged as private for security considerations.
- `kwargs_public` &mdash; This is a dictionary (`Dict`) that holds a collection of key-value pairs used to represent settings or configurations deemed public. These pairs are subsequently passed as parameters to the underlying `kafka.KafkaProducer()` constructor. It defaults to `None`.
- `kwargs_private` &mdash; This dictionary (`Dict`) is used to store key-value pairs, typically representing configurations that are of a private or sensitive nature. These pairs are subsequently passed as parameters to the underlying `kafka.KafkaProducer()` constructor. It defaults to `None`.

### Redis data store profile

```python
profile = DatastoreProfileRedis(name="test_profile", endpoint_url="redis://11.22.33.44:6379", username="user", password="password")
RedisNoSqlTarget(path="ds://test_profile/a/b")
```

### S3 data store profile


```
profile = DatastoreProfileS3(name="test_profile")
ParquetTarget(path="ds://test_profile/aws_bucket/path/to/parquet.pq")
```

`DatastoreProfileS3` init parameters:
- `name` &mdash; Name of the profile
- `endpoint_url` &mdash; A string representing the endpoint URL of the S3 service. It's typically required for non-AWS S3-compatible services. If not provided, the default is `None`. The equivalent to this parameter in environment authentication is env["S3_ENDPOINT_URL"].
- `force_non_anonymous` &mdash; A string that determines whether to force non-anonymous access to the S3 bucket. The default value is `None`, meaning the behavior is not explicitly set. The equivalent to this parameter in environment authentication is - `force_non_anonymous` &mdash; A string that determines whether to force non-anonymous access to the S3 bucket. The default value is `None`, meaning the behavior is not explicitly set. The equivalent to this parameter in environment authentication is env["S3_NON_ANONYMOUS"].
- `profile_name` &mdash; A string representing the name of the profile. This might be used to refer to specific named configurations for connecting to S3. The default value is `None`. The equivalent to this parameter in environment authentication is env["AWS_PROFILE"].
- `assume_role_arn` &mdash; A string representing the Amazon Resource Name (ARN) of the role to assume when interacting with the S3 service. This can be useful for granting temporary permissions. By default, it is set to `None`. The equivalent to this parameter in environment authentication is 
- `access_key` &mdash; A string representing the access key used for authentication to the S3 service. It's one of the credentials parts when you're not using anonymous access or IAM roles. For privacy reasons, it's tagged as a private attribute, and its default value is `None`. The equivalent to this parameter in environment authentication is env["MLRUN_AWS_ROLE_ARN"].
- `secret_key` &mdash; A string representing the secret key, which pairs with the access key, used for authentication to the S3 service. It's the second part of the credentials when not using anonymous access or IAM roles. It's also tagged as private for privacy and security reasons. The default value is `None`. The equivalent to this parameter in environment authentication is env["AWS_SECRET_ACCESS_KEY"].






### See also
- {py:class}`~mlrun.projects.MlrunProject.list_datastore_profiles` 
- {py:class}`~mlrun.projects.MlrunProject.get_datastore_profile`
- {py:class}`~mlrun.datastore.datastore_profile.register_temporary_client_datastore_profile` 
- {py:class}`~mlrun.projects.MlrunProject.delete_datastore_profile`

The methods `get_datastore_profile()` and `list_datastore_profiles()` only return public information about 
the profiles. Access to private attributes is restricted to applications running in Kubernetes pods.



