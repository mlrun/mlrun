# Installation Guide <!-- omit in toc -->

This guide outlines the steps for installing and running MLRun locally.

> **Note:** These instructions use `mlrun` as the namespace (`-n` parameter). You may want to choose a different namespace in your kubernetes cluster.

- [Run MLRun on a Local Docker Registry](#run-mlrun-on-a-local-docker-registry)
- [Install MLRun on a Kubernetes Cluster](#install-mlrun-on-a-kubernetes-cluster)
  - [Install Shared Volume Storage (an NFS Server Provisioner)](#install-shared-volume-storage-an-nfs-server-provisioner)
  - [Install the MLRun API and Dashboard (UI) Services](#install-the-mlrun-api-and-dashboard-ui-services)
  - [Install a Jupyter Server with a Preloaded MLRun Package.](#install-a-jupyter-server-with-a-preloaded-mlrun-package)
  - [Start Working](#start-working)

<a id="local-docker"></a>
## Run MLRun on a Local Docker Registry

To use MLRun with your local Docker registry, run the MLRun API service, dashboard, and example Jupyter server by using the following script.

> **Note:**
> - By default the MLRun API service will run inside the Jupyter server, set the MLRUN_DBPATH env var in Jupyter to point to an alternative service address.
> - The artifacts and DB will be stored under **/home/jovyan/data**, use docker -v option to persist the content on the host (e.g. `-v $(SHARED_DIR}:/home/jovyan/data`)
> - Using Docker is limited to local runtimes.

```sh
MLRUN_IP=localhost
SHARED_DIR=/home/me/data
# On Windows, use host.docker.internal for MLRUN_IP

docker pull quay.io/iguazio/mlrun-ui:latest
docker pull mlrun/jupy:latest

docker run -it -p 4000:80 --rm -d --name mlrun-ui -e MLRUN_API_PROXY_URL=http://${MLRUN_IP}:8080 quay.io/iguazio/mlrun-ui:latest
docker run -it -p 8080:8080 -p 8888:8888 --rm -d --name jupy -v $(SHARED_DIR}:/home/jovyan/data mlrun/jupy:latest
```

When the execution completes &mdash;

- Open Jupyter Notebook on port 8888 and run the code in the [**examples/mlrun_basics.ipynb**](https://github.com/mlrun/mlrun/blob/master/examples/mlrun_basics.ipynb) notebook.
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

<a id="k8s-install-start-working"></a>
### Start Working

- Open Jupyter Notebook on NodePort `30040` and run the code in the [**examples/mlrun_basics.ipynb**](https://github.com/mlrun/mlrun/blob/master/examples/mlrun_basics.ipynb) notebook.
- Use the dashboard at NodePort `30068`.

> **Note:**
> - You can change the ports by editing the YAML files.
> - You can select to use a Kubernetes Ingress for better security.

