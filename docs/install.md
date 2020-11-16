# Installation Guide <!-- omit in toc -->

This guide outlines the steps for installing and running MLRun locally.

> **Note:** These instructions use `mlrun` as the namespace (`-n` parameter). You may want to choose a different namespace in your kubernetes cluster.

- [Run MLRun on a Local Docker Registry](#run-mlrun-on-a-local-docker-registry)
- [Install MLRun on a Kubernetes Cluster](#install-mlrun-on-a-kubernetes-cluster)
  - [Create a namespace](#create-a-namespace)
  - [Install a Shared Volume Storage](#install-a-shared-volume-storage)
    - [NFS Server Provisioner](#nfs-server-provisioner)
  - [Install the MLRun API and Dashboard (UI) Services](#install-the-mlrun-api-and-dashboard-ui-services)
  - [Install a Jupyter Server with a Preloaded MLRun Package.](#install-a-jupyter-server-with-a-preloaded-mlrun-package)
  - [Install Kubeflow](#install-kubeflow)
  - [Start Working](#start-working)

<a id="local-docker"></a>
## Run MLRun on a Local Docker Registry

To use MLRun with your local Docker registry, run the MLRun API service, dashboard, and example Jupyter server by using the following script.

> **Note:**
> - By default the MLRun API service will run inside the Jupyter server, set the MLRUN_DBPATH env var in Jupyter to point to an alternative service address.
> - The artifacts and DB will be stored under **/home/jovyan/data**, use docker -v option to persist the content on the host (e.g. `-v $(SHARED_DIR}:/home/jovyan/data`)
> - Using Docker is limited to local runtimes.

```sh
SHARED_DIR=~/mlrun-data

docker pull mlrun/jupyter:0.5.3
docker pull mlrun/mlrun-ui:0.5.3

docker network create mlrun-network
docker run -it -p 8080:8080 -p 8888:8888 --rm -d --network mlrun-network --name jupyter -v ${SHARED_DIR}:/home/jovyan/data mlrun/jupyter:0.5.3
docker run -it -p 4000:80 --rm -d --network mlrun-network --name mlrun-ui -e MLRUN_API_PROXY_URL=http://jupyter:8080 mlrun/mlrun-ui:0.5.3
```

When the execution completes &mdash;

- Open Jupyter Notebook on port 8888 and run the code in the [**examples/mlrun_basics.ipynb**](https://github.com/mlrun/mlrun/blob/master/examples/mlrun_basics.ipynb) notebook.
- Use the MLRun dashboard on port 4000.

<a id="k8s-cluster"></a>
## Install MLRun on a Kubernetes Cluster

Perform the following steps to install and run MLRun on a Kubernetes cluster.
> **Note:** The outlined procedure allows using the local, job, and Dask runtimes.
> To use the MPIJob (Horovod) or Spark runtimes, you need to install additional custom resource definitions (CRDs).

- [Create a namespace](#k8s-create-a-namespace)
- [Install a shared volume storage](#k8s-install-a-shared-volume-storage)
- [Install the MLRun API and dashboard (UI) services](#k8s-install-mlrun-api-n-ui-services)

<a id=k8s-create-a-namespace></a>
### Create a namespace

Create a namespace for mlrun. For example:

``` sh
kubectl create namespace mlrun
```

<a id="k8s-install-a-shared-volume-storage"></a>
### Install a Shared Volume Storage

You can use any shared file system (or object storage, with some limitations) for sharing artifacts and/or code across containers.

To store data on your Kubernetes cluster itself, you will need to define a [**persistent volume**](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)

#### NFS Server Provisioner
The following example uses a shared NFS server and a Helm chart for the installation:

1. Run the following commands (provided Helm is installed):
    ```sh
    helm repo add stable https://kubernetes-charts.storage.googleapis.com/
    helm install -n mlrun nfsprov stable/nfs-server-provisioner
    ```
2. Create a [**PersistentVolumeClaim**](https://raw.githubusercontent.com/mlrun/mlrun/master/hack/local/nfs-pvc.yaml) (PVC) for a shared NFS volume by running the following command:
    ```sh
    kubectl apply -n mlrun -f https://raw.githubusercontent.com/mlrun/mlrun/master/hack/local/nfs-pvc.yaml
    ```

<a id="k8s-install-mlrun-api-n-ui-services"></a>
### Install the MLRun API and Dashboard (UI) Services

If you plan to push containers or use a private registry, you need to first create a secret with your Docker registry information.
You can do this by running the following command:
```sh
kubectl create -n mlrun secret docker-registry my-docker --docker-server=https://index.docker.io/v1/ --docker-username=<your-user> --docker-password=<your-password> --docker-email=<your-email>
```

Run the following command to apply [**mlrun-local.yaml**](https://raw.githubusercontent.com/mlrun/mlrun/master/hack/local/mlrun-local.yaml):
```sh
kubectl apply -n mlrun -f https://raw.githubusercontent.com/mlrun/mlrun/master/hack/local/mlrun-local.yaml
```

<a id="k8s-install-jupyter-service-w-mlrun"></a>
### Install a Jupyter Server with a Preloaded MLRun Package.

Run the following command to apply [**mljupy.yaml**](https://raw.githubusercontent.com/mlrun/mlrun/master/hack/local/mljupy.yaml):
```sh
kubectl apply -n mlrun -f https://raw.githubusercontent.com/mlrun/mlrun/master/hack/local/mljupy.yaml
```

To change or add packages, see the Jupyter Dockerfile ([**Dockerfile.jupy**](https://github.com/mlrun/mlrun/blob/master/hack/local/Dockerfile.jupy)).

### Install Kubeflow

MLRun enables you to run your functions while saving outputs and artifacts in a way that is visible to Kubeflow Pipelines. If you wish to use this capability you will need to install Kubeflow on your cluster. Refer to the [**Kubeflow documentation**](https://www.kubeflow.org/docs/started/getting-started/) for more information.

<a id="k8s-install-start-working"></a>
### Start Working

- Open Jupyter Notebook on NodePort `30040` and run the code in the [**examples/mlrun_basics.ipynb**](https://github.com/mlrun/mlrun/blob/master/examples/mlrun_basics.ipynb) notebook.
- Use the dashboard at NodePort `30068`.

> **Note:**
> - You can change the ports by editing the YAML files.
> - You can select to use a Kubernetes Ingress for better security.

