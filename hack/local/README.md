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
> - You must provide valid paths for the shared data directory and the local host IP.
> - Both the Jupyter and MLRun services use the path **/home/jovyan/data** within the shared directory to reference the data.
> - Using Docker is limited to local runtimes.

```
SHARED_DIR=/home/me/data
LOCAL_IP=x.x.x.x
# On Windows, use host.docker.internal for LOCAL_IP

docker pull quay.io/iguazio/mlrun-ui:latest
docker pull mlrun/mlrun-api:latest
docker pull mlrun/jupy:latest

docker run -it -p 8080:8080 --rm -d --name mlrun-api -v $(SHARED_DIR}:/home/jovyan/data -e MLRUN_HTTPDB_DATA_VOLUME=/home/jovyan/data mlrun/mlrun-api:0.4.5
docker run -it -p 4000:80 --rm -d --name mlrun-ui -e MLRUN_API_PROXY_URL=http://${LOCAL_IP}:8080 quay.io/iguazio/mlrun-ui:latest
docker run -it -p 8888:8888 --rm -d --name jupy -v $(SHARED_DIR}:/home/jovyan/data -e MLRUN_DBPATH=http://${LOCAL_IP}:8080 -e MLRUN_ARTIFACT_PATH=/home/jovyan/data mlrun/jupy:latest
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
    helm install stable/nfs-server-provisioner --name nfsprov
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

Copy the [**mlrun-local.yaml**](mlrun-local.yaml) file to your cluster, edit it as needed, and run the following command from the directory that contains the file:
```sh
kubectl apply -n <namespace> -f mlrun-local.yaml
```

<a id="k8s-install-jupyter-service-w-mlrun"></a>
### Install a Jupyter Server with a Preloaded MLRun Package

Copy the [**mljupy.yaml**](mljupy.yaml) file to you cluster and run the following command from the directory that contains the file:
```sh
kubectl apply -n <namespace> -f mljupy.yaml
```

To change or add packages, see the Jupyter Dockerfile ([**Dockerfile.jupy**](Dockerfile.jupy)).

<a id="k8s-install-start-working"></a>
### Start Working

- Open Jupyter Notebook on NodePort `30040` and run the code in the [**examples/mlrun_basics.ipynb**](/examples/mlrun_basics.ipynb) notebook.
- Use the dashboard at NodePort `30068`.

> **Note:**
> - You can change the ports by editing the YAML files.
> - You can select to use a Kubernetes Ingress for better security.

