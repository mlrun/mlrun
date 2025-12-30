# Storing artifacts in AWS S3 storage

MLRun CE uses a MinIO service as a shared storage for artifacts, and accesses it using S3 protocol. This means that
any path that begins with `s3://` is automatically directed by MLRun to the MinIO service. The default artifact
path is also configured as `s3://mlrun/projects/{{run.project}}/artifacts` which is a path on the `mlrun` bucket in the
MinIO service.

To store artifacts in AWS S3 buckets instead of the local MinIO service, these configurations need to be overridden to 
make `s3://` paths lead to AWS buckets instead.

```{admonition} Note
These configurations are only required for AWS S3 storage, due to the usage of the same S3 protocol in MinIO. For other
storage options (such as GCS, Azure blobs etc.) just modify the artifact path and provide credentials.
```

## Setting up S3 credentials and endpoint

Set up the following project-secrets (refer to [**Data stores**](../store/datastore.md) and [**Project secrets**](../secrets.md#mlrun-managed-secrets)) 
for any project used:

* `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` &mdash; S3 credentials
* `AWS_ENDPOINT_URL_S3` &mdash; the AWS S3 endpoint to use, depending on the region. For example: 
    ``` console
    AWS_ENDPOINT_URL_S3 = https://s3.us-east-2.amazonaws.com/
    ```
    **Note**: `S3_ENDPOINT_URL` is deprecated as of v1.10.0 and will be removed in v1.12.0. Use `AWS_ENDPOINT_URL_S3` instead.

## Disabling auto-mount

Before running any MLRun job that writes to an S3 bucket, make sure auto-mount is disabled for it, since by default
auto-mount adds S3 configurations that point at the MinIO service (refer to 
[**Function storage**](../runtimes/function-storage.md) for more details on auto-mount). This can be done in one
of following ways:

* Set the client-side MLRun configuration to disable auto-mount. This disables auto-mount for any functions you subsequently run:
    ```python
    from mlrun.config import config as mlconf

    mlconf.storage.auto_mount_type = "none"
    ```
* If running MLRun from an IDE, the configuration can be overridden using an environment variable. Set the following
  environment variable for your IDE environment:
    ```python
    MLRUN_STORAGE__AUTO_MOUNT_TYPE = "none"
    ```
* Disable auto-mount for a specific function. This must be done before running the function for the first time:
    ```python
    function.spec.disable_auto_mount = True
    ```

## Changing the artifact path

The artifact path needs to be modified since the bucket name is set to `mlrun` by default. It is recommended to keep 
the same path structure as the default, while modifying the bucket name. For example:
```text
s3://<bucket name>/projects/{{run.project}}/artifacts
```

The artifact path can be set in several ways, refer to [**Artifact path**](../store/artifacts.md#artifact-path) 
for more details.