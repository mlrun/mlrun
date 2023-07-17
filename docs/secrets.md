(secrets)=
# Working with secrets<!-- omit in toc -->
When executing jobs through MLRun, the code might need access to specific secrets, for example to access data 
residing on a data-store that requires credentials (such as a private S3 bucket), or many other similar needs.

MLRun provides some facilities that allow handling secrets and passing those secrets to execution jobs. It's 
important to understand how these facilities work, as this has implications on the level of security they provide
and how much exposure they create for your secrets.

**In this section**
- [Overview](#overview)
- [MLRun-managed secrets](#mlrun-managed-secrets)
  - [Using tasks with secrets](#using-tasks-with-secrets)
  - [Secret providers](#secret-providers)
    - [Kubernetes project secrets](#kubernetes-project-secrets)
    - [Azure Vault](#azure-vault)
    - [Demo/Development secret providers](#demo-development-secret-providers)
- [Externally managed secrets](#externally-managed-secrets)
  - [Mapping secrets to environment](#mapping-secrets-to-environment)
  - [Mapping secrets as files](#mapping-secrets-as-files)

## Overview
There are two main use-cases for providing secrets to an MLRun job. These are:

- [Use MLRun-managed secrets](#mlrun-managed-secrets). This is a flow that enables the MLRun user (for example a 
data scientist or engineer) to create and use secrets through interfaces that MLRun implements and manages.
- [Create secrets externally](#externally-managed-secrets) to MLRun using a Kubernetes secret or some other secret 
management framework (such as Azure vault), and utilize these secrets from within MLRun to enrich execution jobs. For 
example, the secrets are created and managed by an IT admin, and the data-scientist only accesses them.

The following sections cover the details of those two use-cases.

## MLRun-managed secrets
The easiest way to pass secrets to MLRun jobs is through the MLRun project secrets mechanism. MLRun jobs automatically
gain access to all project secrets defined for the same project. More details are available 
[later in this page](#kubernetes-project-secrets). 

The following is an example of using project secrets:

```python
# Create project secrets for the myproj project
project = mlrun.get_or_create_project("myproj", "./")
secrets = {'AWS_KEY': '111222333'}
project.set_secrets(secrets=secrets, provider="kubernetes")

# Create and run the MLRun job
function = mlrun.code_to_function(
    name="secret_func",
    filename="my_code.py",
    handler="test_function",
    kind="job",
    image="mlrun/mlrun"
)
function.run()
```

The handler defined in `my_code.py` accesses the `AWS_KEY` secret by using the 
{py:func}`~mlrun.execution.MLClientCtx.get_secret()` API:
```python
def test_function(context):
    context.logger.info("running function")
    aws_key = context.get_secret("AWS_KEY")
    # Use aws_key to perform processing.
    ...
```

To create **GIT_TOKEN** secrets, use this command:
```
project.set_secrets({"GIT_TOKEN":<git token>}
```

### Using tasks with secrets
MLRun uses the concept of tasks to encapsulate runtime parameters. Tasks are used to specify execution context
such as hyper-parameters. They can also be used to pass details about secrets that are going to be used in the 
runtime. This allows for control over specific secrets passed to runtimes, and support for the various MLRun secret
providers.

To pass secret parameters, use the Task's {py:func}`~mlrun.model.RunTemplate.with_secrets()` function. For example, 
the following command passes specific project-secrets to the execution context:

```{code-block} python
:emphasize-lines: 8-8

function = mlrun.code_to_function(
    name="secret_func",
    filename="my_code.py",
    handler="test_function",
    kind="job",
    image="mlrun/mlrun"
)
task = mlrun.new_task().with_secrets("kubernetes", ["AWS_KEY", "DB_PASSWORD"])
run = function.run(task, ...)
```

The {py:func}`~mlrun.model.RunTemplate.with_secrets()` function tells MLRun what secrets the executed code needs to 
access. The MLRun framework prepares the needed infrastructure to make these secrets available to the runtime, 
and passes information about them to the execution framework by specifying those secrets in the spec of the runtime. 
For example, if running a kubernetes job, the secret keys are noted in the generated pod's spec.

The actual details of MLRun's handling of the secrets differ per the **secret provider** used. The following sections
provide more details on these providers and how they handle secrets and their values.

Regardless of the type of secret provider used, the executed code uses the 
{py:func}`~mlrun.execution.MLClientCtx.get_secret()` API to gain access to the value of the secrets passed to it, 
as shown in the above example.

### Secret providers
MLRun provides several secret providers. Each of these providers functions differently and 
have different traits with respect to what secrets can be passed and how they're handled. It's important to understand 
these parameters to make sure secrets are not compromised and that their secrecy is maintained.

```{warning}
The [Inline](#inline), [environment](#environment) and [file](#file) providers do not guarantee 
confidentiality of the secret values handled by them, and **should only be used for development and demo purposes**. 
The [Kubernetes](#kubernetes-project-secrets) and [Azure Vault](#azure-vault) providers are secure and should be used 
for any other use-case.
```

#### Kubernetes project secrets
MLRun can use Kubernetes (k8s) secrets to store and retrieve secret values on a per-project basis. This method
is supported for all runtimes that generate k8s pods.  MLRun creates a k8s secret per project, and stores 
multiple secret keys within this secret. Project secrets can be created through the MLRun SDK as well as 
through the MLRun UI. 

By default, all jobs in a project automatically get access to all the associated project secrets. There is
no need to use ```with_secrets``` to provide access to project secrets.

##### Creating project secrets
To populate the MLRun k8s project secret with secret values, use the project object's 
{py:func}`~mlrun.projects.MlrunProject.set_secrets` function, which accepts a dictionary of secret values or
a file containing a list of secrets. For example:

```python
# Create project secrets for the myproj project.
project = mlrun.get_or_create_project("myproj", "./")
secrets = {'password': 'myPassw0rd', 'AWS_KEY': '111222333'}
project.set_secrets(secrets=secrets, provider="kubernetes")
```

```{warning}
This action should not be part of the code committed to `git` or part of ongoing execution - it is only a setup 
action, which normally should only be executed once. After the secrets are populated, this code should be removed 
to protect the confidentiality of the secret values.
```

The MLRun API does not allow the user to see project secrets values, but it does allow 
seeing the keys that belong to a given project, assuming the user has permissions on that specific project. 
See the {py:class}`~mlrun.db.httpdb.HTTPRunDB` class documentation for additional details.

When MLRun is executed in the Iguazio platform, the secret management APIs are protected by the platform such
that only users with permissions to access and modify a specific project can alter its secrets.

##### Creating secrets in the Projects UI page
The Settings dialog in the Projects page, accessed with the Settings icon, has a Secrets tab where you can 
add secrets as key-value pairs. The secrets are automatically available to all jobs belonging to this project. 
Users with the Editor or Admin role can add, modify, and delete secrets, and assign new secret values. 
Viewers can only view the secret keys. The values themselves are not visible to any users.

##### Accessing the secrets
By default, any runtime not executed locally (`local=False`) automatically gains access to all the secrets of the project it 
belongs to, so no configuration is required to enable that. 
**Jobs that are executed locally (`local=True`) do not have access to the project secrets.**
It is possible to limit access of an executing job to a 
subset of these secrets by calling the following function with a list of the secrets to be accessed:

```python
task.with_secrets('kubernetes', ['password', 'AWS_KEY'])
```

When the job is executed, the MLRun framework adds environment variables to the pod spec whose value is retrieved 
through the k8s `valueFrom` option, with `secretKeyRef` pointing at the secret maintained by MLRun.
As a result, this method does not expose the secret values at all, except inside the pod executing the code where
the secret value is exposed through an environment variable. This means that even a user with `kubectl` looking at the 
pod spec cannot see the secret values. 

Users, however, can view the secrets using the following methods:

-  Run `kubectl` to view the actual contents of the k8s secret.
-  Perform `kubectl exec` into the running pod, and examine the environment variables.

To maintain the confidentiality of secret values, these operations must be strictly limited across the system by using 
k8s RBAC and ensuring that elevated permissions are granted to a very limited number of users (very few users have and 
use elevated permissions).

##### Accessing secrets in nuclio functions

Nuclio functions do not have the MLRun context available to retrieve secret values. Secret values need to be retrieved 
from the environment variable of the same name. For example, to access the `AWS_KEY` secret in a nuclio function use:
```python
aws_key = os.environ.get("AWS_KEY")
```

#### Azure Vault
MLRun can serve secrets from an Azure key Vault. 

```{Note}
Azure key Vaults support 3 types of entities - `keys`, `secrets` and `certificates`. MLRun only supports accessing 
`secret` entities.
```

##### Setting up access to Azure key vault
To enable this functionality, a secret must first be created in the k8s cluster that contains the Azure key Vault 
credentials. This secret should include credentials providing access to your specific Azure key Vault. 
To configure this, the following steps are needed:

1. Set up a key vault in your Azure subscription.
2. Create a service principal in Azure that will be granted access to the key vault. For creating a service principal 
   through the Azure portal follow the steps listed in [this page]( 
   https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal).
3. Assign a key vault access policy to the service principal, as described in 
   [this page](https://docs.microsoft.com/en-us/azure/key-vault/general/assign-access-policy-portal).
4. Create a secret access key for the service principal, following the steps listed in [this page]( 
   https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal#get-tenant-and-app-id-values-for-signing-in). 
    Make sure you have access to the following three identifiers:
   
    - Directory (tenant) id
    - Application (client) id
    - Secret key

5. Generate a k8s secret with those details. Use the following command:

    ```shell
    kubectl -n <namespace> create secret generic <azure_key_vault_k8s_secret> \
       --from-literal=secret=<secret key> \
       --from-literal=tenant_id=<tenant id> \
       --from-literal=client_id=<client id>
    ```

```{note}
The names of the secret keys *must* be as shown in the above example, as MLRun queries them by these exact names.
```

##### Accessing Azure key vault secrets
Once these steps are done, use `with_secrets` in the following manner:

```python
task.with_secrets(
    "azure_vault",
    {
        "name": <azure_key_vault_name>,
        "k8s_secret": <azure_key_vault_k8s_secret>,
        "secrets": [],
    },
)
```

The `name` parameter should point at your Azure key Vault name. The `secrets` parameter is a list of the secret keys
to be accessed from that specific vault. If it's empty (as in the example above) then all secrets in the vault can be
accessed by their key name.

For example, if the Azure Vault has a secret whose name is `MY_AZURE_SECRET` and using the above example for
{py:func}`~mlrun.model.RunTemplate.with_secrets()`, the executed code can use the following statement to access 
this secret:

```python
azure_secret = context.get_secret("MY_AZURE_SECRET")
```

In terms of confidentiality, the executed pod has the Azure secret provided by the user mounted to it. This means
that the access-keys to the vault are visible to a user that `exec`s into the pod in question. The same security
rules should be followed as described in the [Kubernetes](#kubernetes-project-secrets) section above. 

#### Demo/Development secret providers
The rest of the MLRun secret providers are not secure by design, and should only be used for demonstration or 
development purposes. 

<details>
<summary>Expand here for additional details.</summary>

##### Inline
The inline secrets provider is a very basic framework that should mostly be used for testing and demos. The secrets 
passed by this framework are exposed in the source code creating the MLRun function, as well as in the function spec, and
in the generated pod specs. To add inline secrets to a job, perform the following:

```python
task.with_secrets("inline", {"MY_SECRET": "12345"})
```

As can be seen, even the client code exposes the secret value. If this is used to pass secrets to a job running in a kubernetes 
pod, the secret is also visible in the pod spec. This means that any user that can run `kubectl` and is permitted 
to view pod specs can also see the secret keys and their values.

##### Environment
Environment variables are similar to the `inline` secrets, but their client-side value is not specified directly in
code but rather is extracted from a client-side environment variable. For example, if running MLRun on a Jupyter 
notebook and there are environment variables named `MY_SECRET` and `ANOTHER_SECRET` on Jupyter, the following code  
passes those secrets to the executed runtime:

```python
task.with_secrets("env", "MY_SECRET, ANOTHER_SECRET")
```

When generating the runtime execution environment (for example, pod for the `job` runtime), MLRun retrieves the
value of the environment variable and places it in the pod spec. This means that a user with `kubectl` capabilities who
can see pod specs can still see the secret values passed in this manner.

##### File
The file provider is used to pass secret values that are stored in a local file. The file needs to be made of 
lines, each containing a secret and its value separated by `=`. For example:

```shell
# secrets.txt
SECRET1=123456
SECRET2=abcdef
```

Use the following command to add these secrets:

```python
task.with_secrets("file", "/path/to/file/secrets.txt")
```

</details>

## Externally managed secrets
MLRun provides facilities to map k8s secrets that were created externally to jobs that are executed. To enable that,
the spec of the runtime that is created should be modified by mounting secrets to it - either as files or as 
environment variables containing specific keys from the secret.

In the following examples, assume a k8s secret called `my-secret` was created in the same k8s namespace where MLRun is running, with two
keys in it - `secret1` and `secret2`.

### Mapping secrets to environment
 The following example adds these two secret keys as environment variables
to an MLRun job:

```{code-block} python
:emphasize-lines: 7-12

function = mlrun.code_to_function(
    name="secret_func",
    handler="test_function",
    ...
)

function.set_env_from_secret(
    "SECRET_ENV_VAR_1", secret="my-secret", secret_key="secret1"
)
function.set_env_from_secret(
    "SECRET_ENV_VAR_2", secret="my-secret", secret_key="secret2"
)
```

This only takes effect for functions executed remotely, as the secret value is injected to the function pod, which does
not exist for functions executed locally.
Within the function code, the secret values will be exposed as regular environment variables, for example:

```{code-block} python
:emphasize-lines: 4-4 

# Function handler
def test_function(context):
    # Getting the value in the secret2 key.
    my_secret_value = os.environ.get("SECRET_ENV_VAR_2")
    ...
```

### Mapping secrets as files
A k8s secret can be mapped as a filesystem folder to the function pod using the {py:func}`~mlrun.platforms.mount_secret`
function:

```python
# Mount all keys in the secret as files under /mnt/secrets
function.apply(mlrun.platforms.mount_secret("my-secret", "/mnt/secrets/"))
```

In our example, the two keys in `my-secret` are created as two files in the function pod, called `/mnt/secrets/secret1` and `/mnt/secrets/secret2`. Reading these
files provide the values. It is possible to limit the keys mounted to the function - see the documentation
of {py:func}`~mlrun.platforms.mount_secret` for more details.
