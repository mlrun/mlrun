# Working with secrets  <!-- omit in toc -->
When executing jobs through MLRun, it is sometimes required to provide the code access to specific secrets. This may 
be needed for example to access data residing on a data-store that requires credentials (such as a private
S3 bucket), or many other similar needs.

MLRun provides some facilities that allow handling secrets and pass those secrets to execution jobs. However, it's 
important to understand how these facilities work, as this has implications on the level of security they provide
and how much exposure they create for your secrets.

## Overview
MLRun uses the concept of Tasks to encapsulate runtime parameters. Tasks are used to specify execution context
such as hyper-parameters, and can also be used to pass details about secrets that are going to be used in the 
runtime.

To pass secret parameters, use the Task's {py:func}`~mlrun.model.RunTemplate.with_secrets()` function. For example, the following command will
pass secrets provided by a kubernetes secret to the execution context (see next sections for a discussion of secret
providers):

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

Within the code in `my_code.py`, the handler can access these secrets by using the 
{py:func}`~mlrun.execution.MLClientCtx.get_secret()` API:

```{code-block} python
:emphasize-lines: 3-3

def test_function(context, db_name):
    context.logger.info("running function")
    db_password = context.get_secret("DB_PASSWORD")
    # Rest of code can use db_password to perform processing.
    ...
```

The {py:func}`~mlrun.model.RunTemplate.with_secrets()` function tells MLRun what secrets the executed code will need 
access to. The MLRun framework prepares the needed infrastructure to make these secrets available to the runtime, 
and passes information about them to the execution framework by specifying those secrets in the spec of the runtime. 
For example, if running a kubernetes job, the secret keys will be noted in the generated pod's spec.

The actual details of MLRun's handling of the secrets differ per the **secret provider** used. The following sections
provide more details on these providers and how they handle secrets and their values.

Regardless of the type of secret provider used, the executed code uses the same 
{py:func}`~mlrun.execution.MLClientCtx.get_secret()` API to gain access to the value of the secrets passed to it, 
as shown in the above example.

## Secret providers
As mentioned, MLRun provides the user with several secret providers. Each of those providers functions differently and 
has different traits with respect to what secrets can be passed and how they're handled. It's important to understand 
these parameters to make sure secrets are not compromised and their secrecy is maintained.

Generally speaking, the [Inline](#inline), [Env](#environment) and [File](#file) providers do not guarantee 
confidentiality of the secret values handled by them, and should only be used for development and demo purposes. 
The [Kubernetes](#kubernetes) and [Azure Vault](#azure-vault) providers are secure and should be used for any 
other use-case.


### Inline
The inline secrets provider is a very basic framework that should mostly be used for testing and demos. The secrets 
passed by this framework are exposed in the source code creating the MLRun function as well as in the function spec and
in the generated pod specs. To add inline secrets to a job, perform the following:

```python
task.with_secrets("inline", {"MY_SECRET": "12345"})
```

As can be seen, even the client code exposes the secret value. If used to pass secrets to a job running in a kubernetes 
pod the secret will also be visible in the pod spec - this means that any user that can run `kubectl` and is permitted 
to view pod specs will see the secret keys as well as their values.

### Environment
Environment variables are similar to the `inline` secrets, but their client-side value is not specified directly in
code but rather is extracted from a client-side environment variable. For example, if running MLRun on a Jupyter 
notebook and there are environment variables named `MY_SECRET` and `ANOTHER_SECRET` on Jupyter, the following code will 
pass those secrets to the executed runtime:

```python
task.with_secrets("env", "MY_SECRET, ANOTHER_SECRET")
```

When generating the runtime execution environment (for example, pod for the `job` runtime), MLRun will retrieve the
value of the environment variable and place it in the pod spec. This means that a user with `kubectl` capabilities who
can see pod specs will still be able to see secret values passed in this manner.

### File
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

In terms of exposure of secret values, this method is the same as for inline or env secrets.

### Kubernetes
MLRun can use a Kubernetes (k8s) secret to store and retrieve secret values as required. This way
of passing secrets is currently supported by the `job` and `serving` runtimes.  The k8s provider creates a
k8s secret per project, and can store multiple secret keys within this secret. 

#### Populating the kubernetes secret
To populate the MLRun k8s secret with secret values, MLRun provides APIs that allow the user to perform operations
on the secrets - this can be done through the {py:class}`~mlrun.db.httpdb.HTTPRunDB` class. For example:

```python
secrets = {'password': 'myPassw0rd', 'aws_key': '111222333'}
mlrun.get_run_db().create_project_secrets(
    "project_name",
    provider=mlrun.api.schemas.SecretProviderName.kubernetes,
    secrets=secrets
)
```

```{warning}
This action should not be part of the code committed to `git` or part of ongoing execution - it is only a setup 
action, which normally should only be executed once. After the secrets are populated, this code should be removed 
to protect the confidentiality of the secret values.
```

The {py:class}`~mlrun.db.httpdb.HTTPRunDB` API does not allow the user to observe secret values, but it does allow 
users to see the keys that belong to a given project, assuming the user has permissions on that specific project. 
See the {py:class}`~mlrun.db.httpdb.HTTPRunDB` class documentation for additional details.

When MLRun is executed in the Iguazio platform, the secret management APIs are protected by the platform such
that only users with permissions to access and modify a specific project can alter its secrets.

#### Accessing the secrets
To provide access to these secrets to an executing job, call the following:

```python
task.with_secrets('kubernetes', ['password', 'aws_key'])
```

Note that only the secret keys are passed in this case, since the values are kept in the k8s secret. When this is done,
the MLRun framework adds environment variables to the pod spec whose value is retrieved through the `valueFrom` option
with `secretKeyRef` pointing at the secret maintained by MLRun.

As a result, this method does not expose the secret values at all, except at the actual pod executing the code where
the secret value is exposed through an environment variable. This means that even a user with `kubectl` looking at pod
spec cannot observe the secret values. 

Still, a user will be able to view the secrets using the following methods:

1. Run `kubectl` to view the actual contents of the k8s secret 
2. Perform `kubectl exec` into the running pod, and examine the environment variables

To maintain the confidentiality of secret values, these operations must be strictly limited across the system by using 
k8s RBAC and ensuring that logging into the k8s nodes as a user with elevated permissions is restricted. 

### Azure Vault
MLRun can serve secrets from an Azure key Vault. Azure key Vaults support 3 types of entities - `keys`, `secrets` and 
`certificates`. MLRun only supports accessing `secret` entities.

#### Setting up access to Azure key vault
To enable this functionality, a secret must first be created in the k8s cluster which contains the Azure key Vault 
credentials. This secret should include credentials providing access to your specific Azure key Vault. 
To configure this, the following steps are needed:

1. Set up a key vault in your Azure subscription
2. Create a service principal in Azure that will be granted access to the key vault. For creating a service principal 
   through the Azure portal follow the steps listed in [this page]( 
   https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal)
3. Assign a key vault access policy to the service principal, as described in 
   [this page](https://docs.microsoft.com/en-us/azure/key-vault/general/assign-access-policy-portal)
4. Create a secret access key for the service principal, following the steps listed in [this page]( 
   https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal#get-tenant-and-app-id-values-for-signing-in). 
    Make sure you have access to the following 3 identifiers:
   
    1. Directory (tenant) id
    2. Application (client) id
    3. Secret key

5. Generate a k8s secret with those details. This can be done using the following command:

    ```shell
    kubectl -n <namespace> create secret generic <azure_key_vault_k8s_secret> \
       --from-literal=secret=<secret key> \
       --from-literal=tenant_id=<tenant id> \
       --from-literal=client_id=<client id>
    ```

```{note}
The names of the secret keys *must* be as shown in the above example, as MLRun queries them by these exact names.
```

#### Accessing Azure key vault secrets
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

In terms of confidentiality, the executed pod will have the Azure secret provided by the user mounted to it. This means
that the access-keys to the vault will be visible to a user that `exec`s into the pod in question. The same security
rules should be followed as described in the [Kubernetes](#kubernetes) section above. 
