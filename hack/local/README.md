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

<a id="local-docker"></a>
## Run MLRun on a Local Docker Registry

To use MLRun with your local Docker registry, run the MLRun API service, dashboard, and example Jupyter server by using the following script.

> **Note:**
> - By default the MLRun API service will run inside the Jupyter server, set the MLRUN_DBPATH env var in Jupyter to point to an alternative service address.
> - The artifacts and DB will be stored under **/home/jovyan/data**, use docker -v option to persist the content on the host (e.g. `-v ${SHARED_DIR}:/home/jovyan/data`)
> - Using Docker is limited to local runtimes.

```
SHARED_DIR=~/mlrun-data

docker pull mlrun/jupyter:0.5.3
docker pull mlrun/mlrun-ui:0.5.3

docker network create mlrun-network
docker run -it -p 8080:8080 -p 8888:8888 --rm -d --network mlrun-network --name jupyter -v ${SHARED_DIR}:/home/jovyan/data mlrun/jupyter:0.5.3
docker run -it -p 4000:80 --rm -d --network mlrun-network --name mlrun-ui -e MLRUN_API_PROXY_URL=http://jupyter:8080 mlrun/mlrun-ui:0.5.3
```

When the execution completes &mdash;

- Open Jupyter Notebook on port 8888 and run the code in the [**examples/mlrun_basics.ipynb**](/examples/mlrun_basics.ipynb) notebook.
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
kubectl create -n <namespace> secret docker-registry my-docker --docker-server=https://index.docker.io/v1/ --docker-username=<your-user> --docker-password=<your-password> --docker-email=<your-email>
```

Copy the [**mlrun-local.yaml**](mlrun-local.yaml) file to your cluster, edit the registry and other attributes as needed, for example:

```yaml
    - name: DEFAULT_DOCKER_REGISTRY
      value: "https://index.docker.io/v1/"
    - name: DEFAULT_DOCKER_SECRET
      value: my-docker
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

- Open Jupyter Notebook on NodePort `30040` and run the code in the [**examples/mlrun_basics.ipynb**](/examples/mlrun_basics.ipynb) notebook.
- Use the dashboard at NodePort `30068`.

> **Note:**
> - You can change the ports by editing the YAML files.
> - You can select to use a Kubernetes Ingress for better security.

