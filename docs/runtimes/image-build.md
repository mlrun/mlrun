(build-function-image)=
# Build function image

As discussed in {ref}`images-usage`, MLRun provides pre-built images which contain the components necessary to execute
an MLRun runtime. In some cases, however, custom images need to be created. 
This page details this process and the available options.

## When is a build required?

In many cases an MLRun runtime can be executed without having to build an image. This will be true when
the basic MLRun images fulfill all the requirements for the code to execute. It is required to build an image 
if one of the following is true:

- The code uses additional Python packages, OS packages, scripts or other configurations that need to be applied
- The code uses different base-images or different versions of MLRun images than provided by default
- Executed source code has changed, and the image has the code packaged in it - see
  [here](mlrun_jobs.html#deploy-build-the-function-container) for more details on source code, and using 
  {py:func}`~mlrun.runtimes.BaseRuntime.with_code()` to avoid re-building the image when the code has changed
- The code runs Nuclio functions, which are packaged as images (the build is triggered by MLRun and executed by 
  Nuclio)

The build process in MLRun is based on [Kaniko](https://github.com/GoogleContainerTools/kaniko) and automated by MLRun -
MLRun generates the Dockerfile for the build process, and configures Kaniko with parameters needed for the build.

Building images is done through functions provided by the {py:class}`~mlrun.projects.MlrunProject` class. By using 
project functions, the same process is used to build and deploy a stand-alone function or functions serving as steps 
in a pipeline.

## Automatically building images
MLRun has the capability to auto-detect when a function image needs to first be built. Following is an example that
will require building of the image:

```python
project = mlrun.new_project(project_name, "./proj")

project.set_function(
    "train_code.py",
    name="trainer",
    kind="job",
    image="mlrun/mlrun",
    handler="train_func",
    requirements=["pandas"],
)

# auto_build will trigger building the image before running,
# due to the additional requirements.
project.run_function("trainer", auto_build=True)
```

Using the `auto_build` option is only suitable when the build configuration does not change between runs of the
runtime. For example, if during the development process new requirements were added, the `auto_build` parameter should
not be used, and manual build is needed to re-trigger a build of the image.

In the example above, the `requirements` parameter was used to specify a list of additional Python packages required by
the code. This option directly affects the image build process - each requirement is installed using `pip` as 
part of the docker-build process. The `requirements` parameter can also contain a path to a requirements file, making
it easier to reuse an existing configuration rather than specify a list of packages.


## Manually building an image

To manually build an image, use the {py:func}`~mlrun.projects.build_function()` function, which provides multiple 
options that control and configure the build process.

### Specifying base image
To use an existing image as the base image for building the image, set the image name in the `base_image` parameter.
Note that this image serves as the base (Dockerfile `FROM` property), and should not to be confused with the 
resulting image name, as specified in the `image` parameter.

```python
project.build_function(
    "trainer",
    base_image="myrepo/my_base_image:latest",
)
```

### Running commands
To run arbitrary commands during the image build, pass them in the `commands` parameter of 
{py:func}`~mlrun.projects.build_function()`. For example:

```python
github_repo = "myusername/myrepo.git@mybranch"

project.build_function(
    "trainer",
    base_image="myrepo/base_image:latest",
    commands=[
        "pip install git+https://github.com/" + github_repo,
        "mkdir -p /some/path && chmod 0777 /some/path",
    ],
)
```

These commands are added as `RUN` operations to the Dockerfile generating the image.

### MLRun package deployment
The `with_mlrun` and `mlrun_version_specifier` parameters allow control over the inclusion of the MLRun package in the
build process. Depending on the base-image used for the build, the MLRun package may already be available in which 
case use `with_mlrun=False`. If not specified, MLRun will attempt to detect this situation - if the image used is one 
of the default MLRun images released with MLRun, `with_mlrun` is automatically set to `False`.
If the code execution requires a different version of MLRun than the one used to deploy the function, 
set the `mlrun_version_specifier` to point at the specific version needed. This uses the published MLRun images
of the specified version instead.
For example:

```python
project.build_function("trainer", with_mlrun=True, mlrun_version_specifier="1.0.0")
```

### Working with code repository
As the code matures and evolves, the code will usually be stored in a git code repository.
When the MLRun project is associated with a git repo (see {ref}`create-projects` for details), functions can be added
by calling {py:func}`~mlrun.projects.MlrunProject.set_function()` and setting `with_repo=True`. This indicates that the 
code of the function should be retrieved from the project code repository.

In this case, the entire code repository will be retrieved from git as part of the image-building process, and cloned
into the built image. This is recommended when the function relies on code spread across multiple files and also is 
usually preferred for production code, since it means that the code of the function is stable, and further modifications 
to the code will not cause instability in deployed images.

During the development phase it may be desired to retrieve the code in runtime, rather than re-build the function
image every time the code changes. To enable this, use {py:func}`~mlrun.projects.MlrunProject.set_source()` which
gets a path to the source (can be a git repository or a tar or zip file) and set `pull_at_runtime=True`.

### Using a private Docker registry
By default, images are pushed to the registry configured during MLRun deployment, using the configured registry 
credentials.

To push resulting images to a different registry, specify the registry URL in the `image` parameter. If
the registry requires credentials, create a k8s secret containing these credentials, and pass its name in the 
`secret_name` parameter.

#### Using ECR as a registry
When using ECR as registry, MLRun uses Kaniko's ECR credentials helper, in which case the secret provided should contain
AWS credentials needed to create ECR repositories, as described [here](https://github.com/GoogleContainerTools/kaniko#pushing-to-amazon-ecr).
MLRun detects automatically that the registry is an ECR registry based on its URL and configures Kaniko to
use the ECR helper. For example:

```python
# AWS credentials stored in a k8s secret -
# kubectl create secret generic ecr-credentials --from-file=<path to .aws/credentials>

project.build_function(
    "trainer",
    image="<aws_account_id>.dkr.ecr.us-east-2.amazonaws.com/myrepo/image:v1",
    secret_name="ecr-credentials",
)
```

When using an ECR registry and not providing a secret name, MLRun assumes that an EC2 instance role is used to authorize access to ECR. 
In this case MLRun clears out AWS credentials provided by project-secrets or environment variables (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY) 
from the Kaniko pod used for building the image. Otherwise Kaniko would attempt to use these credentials for ECR access instead of using the 
instance role. This means it's not possible to build an image with both ECR access via instance role and S3 access using a different set of 
credentials. To build this image, the instance role that has access to ECR must have the permissions required to access S3.

#### Using self-signed registry
If you need to build your function and push the resulting container image to an external Docker registry that uses a self-signed SSL certificate,
you can use Kaniko with the `--skip-tls-verify` flag.
When using this flag, Kaniko ignores the SSL certificate verification while pulling base images and/or pushing the final built image to the registry over HTTPS.

```{admonition} Caution
Using the `--skip-tls-verify` flag poses security risks since it bypasses SSL certificate validation.
Only use this flag in trusted environments or with private registries where you are confident in the security of the network connections.
```

To use this flag, pass it in the extra_args parameter, for example:
```python
project.build_function(
    "<function-name>",
    extra_args="--skip-tls-verify",
)
```
#### Best practice for self-signed registries

Add the certificate authority to the trusted list. If you use a certificate that is not signed by a trusted CA, you are doing so at your own risk.


### Build environment variables
It is possible to pass environment variables that will be set in the Kaniko pod that executes the build. This 
may be useful to pass important information needed for the build process. The variables are passed as a dictionary in
the `builder_env` parameter, for example:

```python
project.build_function(
    "<function-name>",
    builder_env={"GIT_TOKEN": token},
)
```

### Extra arguments
It is also possible to pass custom arguments and flags to Kaniko.
The `extra_args` parameter can be utilized in {py:func}`~mlrun.projects.build_image()`, 
{py:func}`~mlrun.projects.build_function()`, or during the deployment of the function. It provides a way to fine-tune 
the Kaniko build process according to your specific needs.

You can provide the `extra_args` as a string in the format of a CLI command line, just as you would when using 
Kaniko directly, for example:

```python
project.build_function(
    "<function-name>",
    extra_args="--build arg GIT_TOKEN=token --skip-tls-verify",
)
```

Note that when building an image in MLRun, project secrets are automatically passed to the builder pod as environment
variables whose name is the secret key.


## Deploying Nuclio functions
When using Nuclio functions, the image build process is done by Nuclio as part of the deployment of the function. 
Most of the configurations mentioned in this page are available for Nuclio functions as well. To deploy a Nuclio 
function, use {py:func}`~mlrun.projects.deploy_function()` instead of using 
{py:func}`~mlrun.projects.build_function()` and {py:func}`~mlrun.projects.run_function()`.

## Creating default Spark runtime images
When using Spark to execute code, either using a Spark service (remote-spark) or the Spark operator, an image is 
required that contains both Spark binaries and dependencies, and MLRun code and dependencies. 
This image is used in the following scenarios:

1. For remote-spark, the image is used to run the initial MLRun code which will submit the Spark job using the 
   remote Spark service
2. For Spark operator, the image is used for both the driver and the executor pods used to execute the Spark job

This image needs to be created any time a new version of Spark or MLRun is being used, to ensure that jobs are executed
with the correct versions of both products.

To prepare this image, MLRun provides the following facilities:

```python
# For remote Spark
from mlrun.runtimes import RemoteSparkRuntime

RemoteSparkRuntime.deploy_default_image()

# For Spark operator
from mlrun.runtimes import Spark3Runtime

Spark3Runtime.deploy_default_image()
```
