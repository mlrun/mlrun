# Installing and Running MLRun Locally

This guide outlines the steps for installing and running MLRun locally.

> **Note:** Replace the `<namespace>` placeholder in all the commands in this guide with your cluster's Kubernetes namespace.

#### In This Document

- [Run MLRun on a Local Docker Registry](#local-docker)
- [Install an Run MLRun on a Kubernetes Cluster](#k8s-cluster)
  - [Install Shared Volume Storage (an NFS Server Provisioner)](#k8s-install-a-shared-volume-storage)
  - [Install the MLRun API and Dashboard (UI) Services](#k8s-install-mlrun-api-n-ui-services)
  - [Install a Jupyter Server with a Preloaded MLRun Package](#k8s-install-jupyter-service-w-mlrun)
  - [Start Working](#k8s-install-start-working)
  - [Running a local setup with Vault support](#local-vault)

<a id="local-docker"></a>
## Run MLRun on a Local Docker Registry

To use MLRun with your local Docker registry, run the MLRun API service, dashboard, and example Jupyter server by using the following script.

> **Note:**
> - Using Docker is limited to local runtimes.
> - By default the MLRun API service will run inside the Jupyter server, set the MLRUN_DBPATH env var in Jupyter to point to an alternative service address.
> - The artifacts and DB will be stored under **/home/jovyan/data**, use docker -v option to persist the content on the host (e.g. `-v ${SHARED_DIR}:/home/jovyan/data`)
> - If Docker is running on Windows with WSL 2, you must create SHARED_DIR before running these commadns. Provide the full path when executing  (e.g. `mkdir /mnt/c/mlrun-data`  `SHARED_DIR=/mnt/c/mlrun-data`)

```
SHARED_DIR=~/mlrun-data

docker pull mlrun/jupyter:1.0.5
docker pull mlrun/mlrun-ui:1.0.5

docker network create mlrun-network
docker run -it -p 8080:8080 -p 8888:8888 --rm -d --network mlrun-network --name jupyter -v ${SHARED_DIR}:/home/jovyan/data mlrun/jupyter:1.0.5
docker run -it -p 4000:80 --rm -d --network mlrun-network --name mlrun-ui -e MLRUN_API_PROXY_URL=http://jupyter:8080 mlrun/mlrun-ui:1.0.5
```

When the execution completes &mdash;

- Open Jupyter Notebook on port 8888 and run the code in the [**mlrun_basics.ipynb**](/examples/mlrun_basics.ipynb) notebook.
- Use the MLRun dashboard on port 4000.

<a id="k8s-cluster"></a>
## Install MLRun on a Kubernetes Cluster

Perform the following steps to install and run MLRun on a Kubernetes cluster.
> **Note:** The outlined procedure allows using the local, job, and Dask runtimes.
> To use the MPIJob (Horovod) or Spark runtimes, you need to install additional custom resource definitions (CRDs).

- [Install shared volume storage (an NFS server provisioner)](#k8s-install-a-shared-volume-storage)
- [Install the MLRun API and dashboard (UI) services](#k8s-install-mlrun-api-n-ui-services)

<a id="k8s-install-a-shared-volume-storage"></a>
### Install Shared Volume Storage (an NFS Server Provisioner)

You can use any shared file system (or object storage, with some limitations) for sharing artifacts and/or code across containers.
The following example uses a shared NFS server and a Helm chart for the installation:

1. Run the following commands (provided Helm is installed):
    ```sh
    helm repo add stable https://kubernetes-charts.storage.googleapis.com/
    helm install nfsprov stable/nfs-server-provisioner
    ```
2. Create a `PersistentVolumeClaim` (PVC) for a shared NFS volume: copy the [**nfs-pvc.yaml**](nfs-pvc.yaml) file to you cluster and run the following command:
    ```sh
    kubectl apply -n <namespace> -f nfs-pvc.yaml
    ```

<a id="k8s-install-mlrun-api-n-ui-services"></a>
### Install the MLRun API and Dashboard (UI) Services

If you plan to push containers or use a private registry, you need to first create a secret with your Docker registry information.
You can do this by running the following command:
```sh
kubectl create -n <namespace> secret docker-registry my-docker-secret --docker-server=https://index.docker.io/v1/ --docker-username=<your-user> --docker-password=<your-password> --docker-email=<your-email>
```

Copy the [**mlrun-local.yaml**](mlrun-local.yaml) file to your cluster, edit the registry and other attributes as needed, for example:

```yaml
    - name: MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY
      value: "default registry url e.g. index.docker.io/<username>, if repository is not set it will default to mlrun"
    - name: MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY_SECRET
      value: my-docker-secret
``` 

and run the following command from the directory that contains the file:
```sh
kubectl apply -n <namespace> -f mlrun-local.yaml
```

<a id="k8s-install-jupyter-service-w-mlrun"></a>
### Install a Jupyter Server with a Preloaded MLRun Package

Copy the [**mljupy.yaml**](mljupy.yaml) file to you cluster and run the following command from the directory that contains the file:
```sh
kubectl apply -n <namespace> -f mljupy.yaml
```

To change or add packages, see the Jupyter Dockerfile ([**Dockerfile.jupy**](dockerfiles/jupyter/Dockerfile)).

<a id="k8s-install-start-working"></a>
### Start Working

- Open Jupyter Notebook on NodePort `30040` and run the code in the [**mlrun_basics.ipynb**](/examples/mlrun_basics.ipynb) notebook.
- Use the dashboard at NodePort `30068`.

> **Note:**
> - You can change the ports by editing the YAML files.
> - You can select to use a Kubernetes Ingress for better security.

<a id="local-vault"></a>
### Running a local setup with Vault support
This section explains how to setup Vault to work with MLRun API and Jupyter pods running in a k8s cluster,
for example on a Minikube cluster running on your laptop or any other k8s deployment.

#### Enabling k8s authentication in Vault
Vault needs to be installed in your environment. This document does not cover the basic instructions on how to do that -
refer to [**Vault documentation**](https://www.vaultproject.io/docs/install) for instructions.

Once Vault is running locally, you need to enable Kubernetes authentication for vault. Follow the instructions in [**Kubernetes auth method**](https://www.vaultproject.io/docs/auth/kubernetes) to enable it. 
This involves setting up a service account with `TokenReview` permissions - follow the instructions in
[**Vault Agent with Kubernetes**](https://learn.hashicorp.com/tutorials/vault/agent-kubernetes) - follow
the steps for service-account creation and configuring Vault to use it.

#### Configuring MLRun API service to work with Vault

> **Note:**
> By default, the MLRun API uses a service-account that does not have permissions to perform list of serviceaccounts. 
> This permission is required for the Vault functionality, so you'll need to manually edit the `mlrun-api` role and
> add `serviceaccounts` to the list of APIs that are permitted.

The MLRun API service needs to have permissions to manipulate Vault k8s roles, Vault policies and of course handle secrets. The Vault authentication
method within MLRun uses a JWT token to authenticate using k8s mode. The JWT token can be placed in one of two locations:

1. If the token is part of the default service-account for the pod, it will be located in the standard path of `/var/run/secrets/kubernetes.io/serviceaccount/token`. 
If this is your chosen approach, then other than setting the API pod's service account properly, nothing else is needed.
    
2. The token can be placed in a location specified in MLRun conf under `secret_stores.vault.token_path` - the default for that is `~/.mlrun/vault`, but can be overridden like any other MLRun configuration by
setting `MLRUN_SECRET_STORES__VAULT__TOKEN_PATH` to the location where you want to mount your token. This approach is handy if you want to utilize a different SA than the 
pod's default SA to authenticate with Vault. In that case, simply extract the token from the SA that you want to use and mount it in the location pointed at by the variable.
  
We will use the 1st approach here - to configure the MLRun API pod, perform the following steps:
1. Set the MLRun Vault URL using the `MLRUN_SECRET_STORES__VAULT__URL` environment variable (mapped to the MLRun config parameter `secret_stores.vault.url`). 
For example, If using Minikube on Mac and you're running Vault on your local laptop then it 
should be set to `http://docker.for.mac.localhost:8200`. If you're running Vault on another
pod then the URL should just be the DNS name of that pod.
2. Set the Vault role for the pod to be `user:mlrun-api`. The role is specified through the `MLRUN_SECRET_STORES__VAULT__ROLE` environment variable
(which maps to the MLRun config parameter `secret_stores.vault.role`).
3. Set a service-account name for the MLRun API pod. By default the service account used is `mlrun-api`, and this is 
the value used in the rest of this example. If you modify the service account name, make sure you modify the rest of the steps accordingly.
4. Create an `mlrun-api-full` Vault policy, with the following content:
    ```yaml
    # Allow access to mlrun-api user-context secrets
    path "secret/data/mlrun/users/mlrun-api" {
      capabilities = ["read", "list", "create", "update", "sudo"]
    }
    
    path "secret/data/mlrun/users/mlrun-api/*" {
      capabilities = ["read", "list", "create", "update", "sudo"]
    }
    
    # Allow access to secrets of all projects
    path "secret/data/mlrun/projects/*" {
      capabilities = ["read", "list", "create", "update"]
    }
    
    # List existing policies
    path "sys/policies/acl"
    {
      capabilities = ["list"]
    }
    
    # Create and manage ACL policies
    path "sys/policies/acl/*"
    {
      capabilities = ["create", "read", "update", "delete", "list", "sudo"]
    }
    
    path "auth/kubernetes/role/*"
    {
      capabilities = ["create", "read", "update", "delete", "list", "sudo"]
    }
    ```
   
   This can be done using the Vault CLI: save the policy content to a file, for example `my-policy.hcl` 
   and use it to create the policy using the command:
   
   `$ vault policy write mlrun-api-full ./my-policy.hcl`

5. Configure a Vault role for the `mlrun-api` user. The role must be named `mlrun-role-user-<username>`.
The role binds together the service account whose JWT token you use, the namespace and the policy to assign
(which determines the permissions you get on Vault). 
Create the role using the following command:

    `$ vault write auth/kubernetes/role/mlrun-role-user-mlrun-api bound_service_account_names=mlrun-api bound_service_account_namespaces=<namespace> policies=mlrun-api-full ttl=12h`

   
#### Configuring Jupyter to work with Vault
Use the same approach as listed above for the MLRun API pod. If you wish to use the same Vault policy and role for this, then you can
create the same role and assign multiple service-accounts to it. For example (using the same example as above):

`$ vault write auth/kubernetes/role/mlrun-role-user-mlrun-api bound_service_account_names=mlrun-api,jupyter bound_service_account_namespaces=<namespace> policies=mlrun-api-full ttl=12h`

Then assign the `jupyter` service-account to the Jupyter pod and you're good to go.
