# Install MLRun locally
 
## Run Using Local Docker

To use MLRun on your local docker run the API service, the UI, 
and the Examples Jupyter server using the following script. 
You must provide a valid path for the shared data dir and the host local IP.

Once its running open Jupyter (in port 8888) and run the `mlrun_basics` notebook in `/examples`.
and open the UI (in port 4000).

 
```
SHARED_DIR=/home/me/data
LOCAL_IP=x.x.x.x
# on windows use host.docker.internal for LOCAL_IP

docker pull quay.io/iguazio/mlrun-ui:latest
docker pull mlrun/mlrun-api:latest
docker pull mlrun/jupy:latest


docker run -it -p 8080:8080 --rm -d --name mlrun-api -v $(SHARED_DIR}:/home/jovyan/data -e MLRUN_HTTPDB_DATA_VOLUME=/home/jovyan/data mlrun/mlrun-api:0.4.5
docker run -it -p 4000:80 --rm -d --name mlrun-ui -e MLRUN_API_PROXY_URL=http://${LOCAL_IP}:8080 quay.io/iguazio/mlrun-ui:latest
docker run -it -p 8888:8888 --rm -d --name jupy -v $(SHARED_DIR}:/home/jovyan/data -e MLRUN_DBPATH=http://${LOCAL_IP}:8080 -e MLRUN_ARTIFACT_PATH=/home/jovyan/data mlrun/jupy:latest

``` 
 <br>
 
## Install on a Local Kubernetes cluster


### Install Shared Volume Storage (NFS Server Provisioner)

You can use any shared file system (or object storage with some limitations) for sharing artifacts and/or code across containers.
Here we will use a shared NFS server and use a helm chart to install it.

Type the following commands (assuming you have installed Helm):

```
helm repo add stable https://kubernetes-charts.storage.googleapis.com/
helm install stable/nfs-server-provisioner --name nfsprov
```

<b> Create a PVC for a shared NFS volume</b>

copy the [nfs-pvc.yaml](nfs-pvc.yaml) file to you cluster and type:

    kubectl apply -n <namespace> -f nfs-pvc.yaml
    
> Note: the `/home/jovyan/data` path is used by both Jupyter and MLRun service to reference the data.
    

### Install MLRun API & UI Services

copy and edit the [mlrun-local.yaml](mlrun-local.yaml) file as needed and type:

    kubectl apply -n <namespace> -f mlrun-local.yaml


### Install a Jupyter Server (with pre-loaded MLRun)

copy the [mljupy.yaml](mljupy.yaml) file to you cluster and type:

    kubectl apply -n <namespace> -f mljupy.yaml
    
See the Jupyter [Dockerfile](Dockerfile.jupy) if you need to change or add packages.

### Start working

login to the Jupyter notebook at Node port `30040` and run the examples 

Use the UI at Nodeport `30068` 

The ports can be changes by editing the yaml files, you can use `ingress` for better security.
    
