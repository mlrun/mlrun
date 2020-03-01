# Install MLRun on a local Kubernetes cluster


### Install NFS Server Provisioner

```
helm repo add stable https://kubernetes-charts.storage.googleapis.com/
helm install stable/nfs-server-provisioner --name nfsprov
```

<b> Create a PVC for a shared NFS volume</b>

copy the [nfs-pvc.yaml](nfs-pvc.yaml) file to you cluster and type:

    kubectl apply -f nfs-pvc.yaml
    

### Install MLRun API & UI Services

copy and edit the [mlrun-local.yaml](mlrun-local.yaml) file as needed and type:

    kubectl apply -f mlrun-local.yaml


### Install a Jupyter Server (with pre-loaded MLRun)

copy the [mljupy.yaml](mljupy.yaml) file to you cluster and type:

    kubectl apply -f mljupy.yaml
    

### Start working

login to the Jupyter notebook at Node port `30040` and run the examples 

Use the UI at Nodeport `30068` 

The ports can be changes by editing the yaml files
    
