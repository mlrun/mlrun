(install-on-kubernetes)=
# Install MLRun CE on Kubernetes

These instructions install the community edition (CE) on your Kubernetes cluster.

```{admonition} Note
These instructions install the community edition {{ ceversion }}, which currently includes the features in MLRun {{ version }}.</br>
```

**In this section**
- [Prerequisites](#prerequisites)
- [Installing the chart](#installing-the-chart)
- [Usage](#usage)
- [Start working](#start-working)
- [Configuring the remote environment](#configuring-the-remote-environment)
- [Uninstalling the chart](#uninstalling-the-chart)
- [Upgrading the chart](#upgrading-the-chart)

## Prerequisites

- Access to a Kubernetes cluster, version >=1.32. To install MLRun on your cluster, you must have administrator permissions. 
For local installation on Windows or Mac, [Docker Desktop](https://www.docker.com/products/docker-desktop) is recommended, see also [Installing on Docker Desktop](#installing-on-docker-desktop) section. 
- The Kubernetes command-line tool (kubectl) compatible with your Kubernetes cluster is installed. Refer to the [kubectl installation 
instructions](https://kubernetes.io/docs/tasks/tools/install-kubectl/) for more information, If you choose to use Docker Desktop, you can use its built-in Kubernetes service, which includes the kubectl CLI out of the box.
- Helm version >=3.16 CLI is installed. Refer to the [Helm installation instructions](https://helm.sh/docs/intro/install/) for more information.
- An accessible Docker Registry (such as [Docker Hub](https://hub.docker.com)). The Registry's URL and credentials are consumed by the applications via a pre-created secret.
- Storage: 
  - 8Gi
  - Set a default storage class for the Kubernetes cluster, in order for the pods to have persistent storage. See the [Kubernetes documentation](https://kubernetes.io/docs/concepts/storage/storage-classes/#storageclass-objects) for more information.
- RAM: A minimum of 8Gi is required for running all the initial MLRun components. The amount of RAM required for running MLRun jobs depends on the job's requirements.
- Review the [MLRun CE installation notes](./mlrun-ce-installation-notes.md) for any additional installation steps you may need to consider.

``` {admonition} Note
The MLRun Community Edition resources are configured initially with the default cluster/namespace resource limits. You can modify the resources from outside if needed.
```

<a id="installing-the-chart"></a>
## Installing the chart

```{admonition} Note
These instructions use `mlrun` as the namespace (`-n` parameter). You can choose a different namespace in your Kubernetes cluster.
```

Create a namespace for the deployed components:

```bash
kubectl create namespace mlrun
```

Add the Community Edition helm chart repo:

```bash
helm repo add mlrun-ce https://mlrun.github.io/ce
```

Run the following command to ensure that the repo is installed and available:
```bash
helm repo list
```

It should output something like:
```bash
NAME        URL
mlrun-ce    https://github.com/mlrun/ce
```

Update the repo to make sure you're getting the latest chart:

```bash
helm repo update
```

Create a secret with your Docker Registry named `registry-credentials`:

```bash
kubectl --namespace mlrun create secret docker-registry registry-credentials \
    --docker-server <your-registry-server> \
    --docker-username <your-username> \
    --docker-password <your-password> \
    --docker-email <your-email>
```
```{admonition} Note
If using Docker hub, the Registry server is `https://registry.hub.docker.com/`. Refer to the [Docker ID documentation](https://docs.docker.com/docker-id/) for 
creating a user with login to configure in the secret.
```

Where:
- `<your-registry-server>` is your Private Docker Registry FQDN. (`index.docker.io/<your-username>` for Docker Hub).
- `<your-username>` is your Docker username.
- `<your-password>` is your Docker password.
- `<your-email>` is your Docker email.

To install the chart with the release name `mlrun-ce` use the following command.  
Note the reference to the pre-created `registry-credentials` secret in `global.registry.secretName`:

```bash
helm --namespace mlrun \
    install mlrun-ce \
    --wait \
    --timeout 960s \
    --set global.registry.url=<registry-url> \
    --set global.registry.secretName=registry-credentials \
    --set global.externalHostAddress=<host-machine-address> \
    mlrun-ce/mlrun-ce
```

Where:
 - `<registry-url>` is the Registry URL that can be authenticated by the `<registry-credentials>` secret (e.g., `index.docker.io/<your-username>` for Docker Hub).
 - `<host-machine-address>` is the IP address of the host machine (or `$(minikube ip)` if using minikube).

When the installation is complete, the helm command prints the URLs and ports of all the MLRun CE services.

```{admonition} Known issue when installing the chart on Macs using Apple silicon (ARM-based architicture):
- The Grafana statistics do not work well in this release. A fix will be delivered in a subsequent release.
```

## Usage

Your applications are now available in your local browser:
- Jupyter Notebook - `http://<host-machine-address>:30040`
- Nuclio - `http://<host-machine-address>:30050`
- MLRun UI - `http://<host-machine-address>:30060`
- MLRun API (external) - `http://<host-machine-address>:30070`
- MinIO API - `http://<host-machine-address>:30080`
- MinIO UI - `http://<host-machine-address>:30090`
- Pipeline UI - `http://<host-machine-address>:30100`
- Grafana UI - `http://<host-machine-address>:30110`


```{admonition} Check state
You can check the current state of the installation via the command `kubectl -n mlrun get pods`, where the main information
is in columns `Ready` and `State`. If all images have already been pulled locally, typically it takes 
a minute for all services to start.
```

```{admonition} Note
You can change the ports by providing values to the helm install command.
You can add and configure a Kubernetes ingress-controller for better security and control over external access.
```

## Start working
    
Open the Jupyter notebook on [**jupyter-notebook UI**](http://localhost:30040) and run the code in the 
[**examples/mlrun_basics.ipynb**](https://github.com/mlrun/mlrun/blob/master/examples/mlrun_basics.ipynb) notebook.

```{admonition} Important
Make sure to save your changes in the `data` folder within the Jupyter Lab. The root folder and any other folders do not retain the changes when you restart the Jupyter Lab.
```

## Configuring the remote environment

You can use your code on a local machine while running your functions on a remote cluster. Refer to [Set up your environment](../setup-guide.md) for more information.

### Installing on Docker Desktop

If you are using Docker Desktop, you can install MLRun CE on your local machine.
Docker Desktop is available for Mac and Windows. For download information, system requirements, and installation instructions, see:

- [Install Docker Desktop on Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
- [Install Docker Desktop on Windows](https://docs.docker.com/docker-for-windows/install/). Note that WSL 2 backend was tested, Hyper-V was not tested.

#### Configuring Docker Desktop

Docker Desktop includes a standalone Kubernetes server and client, as well as Docker CLI integration that runs on your machine. The 
Kubernetes server runs locally within your Docker instance. To enable Kubernetes support and install a standalone instance of Kubernetes 
running as a Docker container, go to **Preferences** > **Kubernetes** and then press **Enable Kubernetes**. Press **Apply & Restart** to 
save the settings and then press **Install** to confirm. This instantiates the images that are required to run the Kubernetes server as 
containers, and installs the `/usr/local/bin/kubectl` command on your machine. For more information, see [the Kubernetes documentation](https://docs.docker.com/desktop/features/kubernetes/).

It's recommended to limit the amount of memory allocated to Kubernetes. If you're using Windows and WSL 2, you can configure global WSL options by placing a `.wslconfig` file into the root directory of 
your users folder: `C:\Users\<yourUserName>\.wslconfig`. Keep in mind that you might need to run `wsl --shutdown` to shut down the WSL 2 VM and then restart your WSL instance for these changes to take effect.

``` console
[wsl2]
memory=8GB # Limits VM memory in WSL 2 to 8 GB
```

To learn about the various UI options and their usage, see:

- [Docker Desktop for Mac user manual](https://docs.docker.com/desktop/setup/install/mac-install/)
- [Docker Desktop for Windows user manual](https://docs.docker.com/desktop/setup/install/windows-install/)

## Uninstalling the chart

The following command deletes the pods, deployments, config maps, services and roles+role bindings associated with the chart and release.

```bash
helm --namespace mlrun uninstall mlrun-ce
```

### Notes on dangling resources
- The created CRDs are not deleted by default and should be manually cleaned up. 
- The created PVs and PVCs are not deleted by default and should be manually cleaned up. 
- As stated above, the images in the Docker Registry are not deleted either and should be cleaned up manually.
- If you installed the chart in its own namespace, it's also possible to delete the entire namespace to clean up all resources (apart from the Docker Registry images).

### Note on terminating pods and hanging resources
This chart generates several persistent volume claims that provide persistency (via PVC) out of the box. 
Upon uninstalling, any hanging / terminating pods hold the PVCs and PVs respectively, as those prevent their safe removal.
Since pods that are stuck in terminating state seem to be a never-ending plague in Kubernetes, note this,
and remember to clean the remaining PVs and PVCs.

### Handing stuck-at-terminating pods
```bash
kubectl --namespace mlrun delete pod --force --grace-period=0 <pod-name>
```

### Reclaim dangling persistency resources:

```{admonition} WARNING 
**This will result in data loss!**
```

```bash
# To list PVCs
$ kubectl --namespace mlrun get pvc
...

# To remove a PVC
$ kubectl --namespace mlrun delete pvc <pvc-name>
...

# To list PVs
$ kubectl --namespace mlrun get pv
...

# To remove a PV
$ kubectl --namespace mlrun delete pv <pv-name>
...
```

## Upgrading the chart

To upgrade to the latest version of the chart, first make sure you have the latest helm repo

```bash
helm repo update
```

Then try to upgrade the chart:

```bash
helm upgrade --install --reuse-values mlrun-ce —namespace mlrun mlrun-ce/mlrun-ce
```

If it fails, reinstall the chart:

1. Remove the current mlrun-ce:
```bash
mkdir ~/tmp
helm get values -n mlrun mlrun-ce > ~/tmp/mlrun-ce-values.yaml #edit the values file if needed
helm uninstall mlrun-ce
```
2.  Reinstall the mlrun-ce, reusing the values:
```bash
helm install -n mlrun --values ~/tmp/mlrun-ce-values.yaml mlrun-ce mlrun-ce/mlrun-ce --devel
```

```{admonition} Note
If your values have fixed mlrun service versions (e.g.: mlrun:1.8.0) then you might want to remove it from the values file to allow newer chart defaults to kick in.
```

