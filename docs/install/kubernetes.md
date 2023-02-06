(install-on-kubernetes)=
# Install MLRun on Kubernetes

**In this section**
- [Prerequisites](#prerequisites)
- [Community Edition Flavors](#community-edition-flavors)
- [Installing the Chart](#installing-the-chart)
- [Configuring Online Feature Store](#configuring-online-feature-store)
- [Usage](#usage)
- [Start working](#start-working)
- [Configuring the remote environment](#configuring-the-remote-environment)
- [Advanced chart configuration](#advanced-chart-configuration)
- [Storage Resources](#storage-resources)
- [Uninstalling the chart](#uninstalling-the-chart)
- [Upgrading the chart](#upgrading-the-chart)

## Prerequisites

- Access to a Kubernetes cluster. You must have administrator permissions in order to install MLRun on your cluster. MLRun fully supports k8s releases 1.22 and 1.23. For local installation 
on Windows or Mac, [Docker Desktop](https://www.docker.com/products/docker-desktop) is recommended. 
- The Kubernetes command-line tool (kubectl) compatible with your Kubernetes cluster is installed. Refer to the [kubectl installation 
instructions](https://kubernetes.io/docs/tasks/tools/install-kubectl/) for more information.
- Helm 3.6 CLI is installed. Refer to the [Helm installation instructions](https://helm.sh/docs/intro/install/) for more information.
- An accessible docker-registry (such as [Docker Hub](https://hub.docker.com)). The registry's URL and credentials are consumed by the applications via a pre-created secret.
- Storage: 
  - 8Gi
  - It is also required to set a default storage class for the kubernetes cluster in order for the pods to have persistent storage. Please see the [Kubernetes documentation](https://kubernetes.io/docs/concepts/storage/storage-classes/#the-storageclass-resource) for more information.
- RAM: A minimum of 8Gi is required for running all the initial MLRun components. The amount of RAM required for running MLRun jobs depends on the job's requirements.

``` {admonition} Note
The MLRun Community Edition resources are configured initially with the default cluster/namespace resources limits. You can modify the resources from outside if needed.
```

## Community Edition flavors

The MLRun CE (Community Edition) includes the following components:
* MLRun - https://github.com/mlrun/mlrun
  - MLRun API
  - MLRun UI
  - MLRun DB (MySQL)
* Nuclio - https://github.com/nuclio/nuclio
* Jupyter - https://github.com/jupyter/notebook (+MLRun integrated)
* MPI Operator - https://github.com/kubeflow/mpi-operator
* Minio - https://github.com/minio/minio/tree/master/helm/minio
* Spark Operator - https://github.com/GoogleCloudPlatform/spark-on-k8s-operator
* Pipelines - https://github.com/kubeflow/pipelines
* Prometheus stack - https://github.com/prometheus-community/helm-charts
  - Prometheus
  - Grafana


<a id="installing-the-chart"></a>
## Installing the Chart

```{admonition} Note
These instructions use `mlrun` as the namespace (`-n` parameter). You can choose a different namespace in your kubernetes cluster.
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
mlrun-ce    https://mlrun.github.io/ce
```

Update the repo to make sure you're getting the latest chart:

```bash
helm repo update
```

Create a secret with your docker-registry named `registry-credentials`:

```bash
kubectl --namespace mlrun create secret docker-registry registry-credentials \
    --docker-server <your-registry-server> \
    --docker-username <your-username> \
    --docker-password <your-password> \
    --docker-email <your-email>
```
> **Note:**
> If using docker hub, the registry server is `https://index.docker.io/v1/`. Refer to the [Docker ID documentation](https://docs.docker.com/docker-id/) for 
> creating a user with login to configure in the secret.

Where:

- `<your-registry-server>` is your Private Docker Registry FQDN. (https://index.docker.io/v1/ for Docker Hub).
- `<your-username>` is your Docker username.
- `<your-password>` is your Docker password.
- `<your-email>` is your Docker email.

```{admonition} Note
First-time MLRun users will experience a relatively longer installation time because all required images 
are being pulled locally for the first time (it will take an average of 10-15 minutes mostly depends on 
your internet speed).
```

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
 - `<registry-url>` is the registry URL that can be authenticated by the `registry-credentials` secret (e.g., `index.docker.io/<your-username>` for Docker Hub).
 - `<host-machine-address>` is the IP address of the host machine (or `$(minikube ip)` if using minikube).

When the installation is complete, the helm command prints the URLs and Ports of all the MLRun CE services.

> **Note:**
> There is currently a known issue with installing the chart on Macs using Apple Silicon (M1). The current pipelines
> mysql database fails to start. The workaround for now is to opt out of pipelines by installing the chart with the 
> `--set pipelines.mysql.enabled=false`. 

## Configuring Online Feature Store
The MLRun Community Edition now supports the online feature store. To enable it, you need to first deploy a REDIS service that is accessible to your MLRun CE cluster.
To deploy a REDIS service, refer to the following [link](https://redis.io/docs/getting-started/).

When you have a REDIS service deployed, you can configure MLRun CE to use it by adding the following helm value configuration to your helm install command:
```bash
--set mlrun.api.extraEnvKeyValue.MLRUN_REDIS__URL=<redis-address>
```

## Usage

Your applications are now available in your local browser:
- jupyter-notebook - `http://<host-machine-address>:30040`
- nuclio - `http://<host-machine-address>:30050`
- mlrun UI - `http://<host-machine-address>:30060`
- mlrun API (external) - `http://<host-machine-address>:30070`
- minio API - `http://<host-machine-address>:30080`
- minio UI - `http://<host-machine-address>:30090`
- pipeline UI - `http://<host-machine-address>:30100`
- grafana UI - `http://<host-machine-address>:30110`


```{admonition} Check state
You can check current state of installation via command `kubectl -n mlrun get pods`, where the main information
is in columns `Ready` and `State`. If all images have already been pulled locally, typically it will take 
a minute for all services to start.
```

```{admonition} Note
You can change the ports by providing values to the helm install command.
You can add and configure a k8s ingress-controller for better security and control over external access.
```

## Start Working
    
Open the Jupyter notebook on [**jupyter-notebook UI**](http://localhost:30040) and run the code in the 
[**examples/mlrun_basics.ipynb**](https://github.com/mlrun/mlrun/blob/master/examples/mlrun_basics.ipynb) notebook.

```{admonition} Important
Make sure to save your changes in the `data` folder within the Jupyter Lab. The root folder and any other folder do not retain the changes when you restart the Jupyter Lab.
```

## Configuring the remote environment

You can use your code on a local machine while running your functions on a remote cluster. Refer to [Set up your remote environment](https://docs.mlrun.org/en/latest/install/remote.html) for more information.

## Advanced chart configuration

Configurable values are documented in the `values.yaml`, and the `values.yaml` of all sub charts. Override those [in the normal methods](https://helm.sh/docs/chart_template_guide/values_files/).

### Opt out of components
The chart installs many components. You might not need them all in your deployment depending on your use cases.
In order to opt out of some of the components, you can use the following helm values:
```bash
...
--set pipelines.enabled=false \
--set kube-prometheus-stack.enabled=false \
--set sparkOperator.enabled=false \
...
```

### Installing on Docker Desktop

If you are using Docker Desktop, you can install MLRun CE on your local machine.
Docker Desktop is available for Mac and Windows. For download information, system requirements, and installation instructions, see:

- [Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/)
- [Install Docker Desktop on Windows](https://docs.docker.com/docker-for-windows/install/). Note that WSL 2 backend was tested, Hyper-V was not tested.

#### Configuring Docker Desktop

Docker Desktop includes a standalone Kubernetes server and client, as well as Docker CLI integration that runs on your machine. The 
Kubernetes server runs locally within your Docker instance. To enable Kubernetes support and install a standalone instance of Kubernetes 
running as a Docker container, go to **Preferences** > **Kubernetes** and then click **Enable Kubernetes**. Click **Apply & Restart** to 
save the settings and then click **Install** to confirm. This instantiates the images that are required to run the Kubernetes server as 
containers, and installs the `/usr/local/bin/kubectl` command on your machine. For more information, see [the Kubernetes documentation](https://docs.docker.com/desktop/kubernetes/).

It's recommended to limit the amount of memory allocated to Kubernetes. If you're using Windows and WSL 2, you can configure global WSL options by placing a `.wslconfig` file into the root directory of your users folder: `C:\Users\<yourUserName>\.wslconfig`. Keep in mind that you might need to run `wsl --shutdown` to shut down the WSL 2 VM and then restart your WSL instance for these changes to take effect.

``` console
[wsl2]
memory=8GB # Limits VM memory in WSL 2 to 8 GB
```

To learn about the various UI options and their usage, see:

- [Docker Desktop for Mac user manual](https://docs.docker.com/docker-for-mac/)
- [Docker Desktop for Windows user manual](https://docs.docker.com/docker-for-windows/)

## Storage resources

When installing the MLRun Community Edition, several storage resources are created:

- **PVs via default configured storage class**: Used to hold the file system of the stacks pods, including the MySQL database of MLRun, Minio for artifacts and Pipelines Storage and more. These are not deleted when the stack is uninstalled to allow upgrades without losing data.
- **Container Images in the configured docker-registry**: When building and deploying MLRun and Nuclio functions via the MLRun Community Edition, the function images are stored in the given configured docker registry. These images persist in the docker registry and are not deleted.

## Uninstalling the Chart

The following command deletes the pods, deployments, config maps, services and roles+role bindings associated with the chart and release.

```bash
helm --namespace mlrun uninstall mlrun-ce
```

### Notes on dangling resources
- The created CRDs are not deleted by default and should be manually cleaned up. 
- The created PVs and PVCs are not deleted by default and should be manually cleaned up. 
- As stated above, the images in the docker registry are not deleted either and should be cleaned up manually.
- If you installed the chart in its own namespace, it's also possible to delete the entire namespace to clean up all resources (apart from the docker registry images).

### Note on terminating pods and hanging resources
This chart generates several persistent volume claims that provide persistency (via PVC) out of the box. 
Upon uninstallation, any hanging / terminating pods will hold the PVCs and PVs respectively, as those prevent their safe removal.
Since pods that are stuck in terminating state seem to be a never-ending plague in k8s, note this,
and remember to clean the remaining PVs and PVCs.

### Handing stuck-at-terminating pods:
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

# To remove a PVC
$ kubectl --namespace mlrun delete pvc <pv-name>
...
```

## Upgrading the chart

In order to upgrade to the latest version of the chart, first make sure you have the latest helm repo

```bash
helm repo update
```

Then upgrade the chart:

```bash
helm upgrade --install --reuse-values mlrun-ce mlrun-ce/mlrun-ce
```