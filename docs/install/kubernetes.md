(install-on-kubernetes)=
# Install MLRun on a Kubernetes Cluster

**In this section**
- [Prerequisites](#prerequisites)
- [Installing on Docker Desktop](#installing-on-docker-desktop)
- [Installing the chart](#installing-the-chart)
- [Installing Kubeflow](#install-kubeflow)
- [Start working](#start-working)
- [Configuring the remote environment](#configuring-the-remote-environment)
- [Advanced chart configuration](#advanced-chart-configuration)
- [Uninstalling the chart](#uninstalling-the-chart)
- [Upgrading the chart](#upgrading-the-chart)

## Prerequisites

- Access to a Kubernetes cluster. You must have administrator permissions in order to install MLRun on your cluster. For local installation 
on Windows or Mac, [Docker Desktop](https://www.docker.com/products/docker-desktop) is recommended. MLRun fully supports k8s releases 1.22 and 1.23.
- The Kubernetes command-line tool (kubectl) compatible with your Kubernetes cluster is installed. Refer to the [kubectl installation 
instructions](https://kubernetes.io/docs/tasks/tools/install-kubectl/) for more information.
- Helm 3.6 CLI is installed. Refer to the [Helm installation instructions](https://helm.sh/docs/intro/install/) for more information.
- An accessible docker-registry (such as [Docker Hub](https://hub.docker.com)). The registry's URL and credentials are consumed by the applications via a pre-created secret.
- Storage: 7Gi

``` {admonition} Note
The MLRun kit resources (MLRun-API, MLRun-UI, Jupyter, and Nuclio) are configured initially with the default cluster/namespace resources limits. You can modify the resources from outside if needed.
```

**Watch the installation:**

<iframe width="187" height="105" src="https://www.youtube.com/embed/Z6mTw7_n8iE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<a id="docker-desktop-installation"></a>
## Installing on Docker Desktop

Docker Desktop is available for Mac and Windows. For download information, system requirements, and installation instructions, see:

- [Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/)
- [Install Docker Desktop on Windows](https://docs.docker.com/docker-for-windows/install/). Note that WSL 2 backend was tested, Hyper-V was not tested.

### Configuring Docker Desktop

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


<a id="installing-the-chart"></a>
## Installing the chart

```{admonition} Note
These instructions use `mlrun` as the namespace (`-n` parameter). You can choose a different namespace in your kubernetes cluster.
```

Create a namespace for the deployed components:

```bash
kubectl create namespace mlrun
```

Add the `v3io-stable` helm chart repo:

```bash
helm repo add v3io-stable https://v3io.github.io/helm-charts/stable
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

To install the chart with the release name `mlrun-kit` use the following command.
Note the reference to the pre-created `registry-credentials` secret in `global.registry.secretName`:

```bash
helm --namespace mlrun \
    install mlrun-kit \
    --wait \
    --timeout 960s \
    --set global.registry.url=<registry-url> \
    --set global.registry.secretName=registry-credentials \
    v3io-stable/mlrun-kit
```

Where `<registry-url>` is the registry URL which can be authenticated by the `registry-credentials` secret (e.g., `index.docker.io/<your-username>` for Docker Hub).

```{admonition} Installing on Minikube/VM**
The Open source MLRun kit uses node ports for simplicity. If your Kubernetes cluster is running inside a VM, 
as is usually the case when using minikube, the Kubernetes services exposed over node ports are not available on 
your local host interface, but instead, on the virtual machine's interface.
To accommodate for this, use the `global.externalHostAddress` value on the chart. For example, if you're using 
the kit inside a minikube cluster (with some non-empty `vm-driver`), pass the VM address in the chart installation 
command as follows:

```bash
helm --namespace mlrun \
    install my-mlrun \
    --wait \
    --timeout 960s \
    --set global.registry.url=<registry URL e.g. index.docker.io/iguazio > \
    --set global.registry.secretName=registry-credentials \
    --set global.externalHostAddress=$(minikube ip) \
    v3io-stable/mlrun-kit
```
Where `$(minikube ip)` shell command resolving the external node address of the k8s node VM.

## Installing Kubeflow

You can run your functions while saving outputs and artifacts in a way that is visible to Kubeflow Pipelines.
To use this capability you need to install Kubeflow on your cluster.
Refer to the [Kubeflow documentation](https://www.kubeflow.org/docs/started/getting-started/) for more information.

## Usage

Your applications are now available in your local browser:

- Jupyter-notebook - http://localhost:30040
- Nuclio - http://localhost:30050
- MLRun UI - http://localhost:30060
- MLRun API (external) - http://localhost:30070 (health check via http://localhost:30070/api/healthz)

```{admonition} Check state
You can check current state of installation via command `kubectl -n mlrun get pods`, where the main information
is in columns `Ready` and `State`. If all images have already been pulled locally, typically it will take 
a minute for all services to start.
```

```{admonition} Note
The above links assume your Kubernetes cluster is exposed on localhost.
If that's not the case, the different components are available on the provided `externalHostAddress`
- You can change the ports by providing values to the helm install command.
- You can add and configure a k8s ingress-controller for better security and control over external access.
```

## Start working

Open Jupyter Lab on [**jupyter-lab UI**](http://localhost:30040) and run the code in [**quick-start.ipynb**](https://github.com/mlrun/mlrun/blob/master/docs/quick-start/quick-start.ipynb) notebook.

```{admonition} Important
Make sure to save your changes in the `data` folder within the Jupyter Lab. The root folder and any other folder do not retain the changes when you restart the Jupyter Lab.
```

## Configuring the remote environment

You can use your code on a local machine while running your functions on a remote cluster.

### Prerequisites

Before you begin, ensure that the following prerequisites are met:

- The MLRun version installed with the MLRun Kit is the same as the MLRun version on your remote cluster. If needed, upgrade MLRun 
either in your local installation or on the remote cluster so that they match.
- You have remote access to your MLRun service (i.e. to the service URL on the remote Kubernetes cluster).

### Setting environment variables

Define your MLRun configuration. 

- As a minimum requirement: Set `MLRUN_DBPATH` to the URL of the remote MLRun database/API service; replace the `<...>` placeholders to identify your remote target:

    ```ini
    MLRUN_DBPATH=<API endpoint of the MLRun APIs service endpoint; e.g., "https://mlrun-api.default-tenant.app.mycluster.iguazio.com">
    ```

- To store the artifacts on the remote server, you need to set the `MLRUN_ARTIFACT_PATH` to the desired root folder of your 
artifact. You can use template values in the artifact path. The supported values are:
   - `{{project}}` to include the project name in the path.
   - `{{run.uid}}` to include the specific run uid in the artifact path. 

   For example:

    ```ini
    MLRUN_ARTIFACT_PATH=/User/artifacts/{{project}}
    ```
    
   or:

    ```ini
    MLRUN_ARTIFACT_PATH=/User/artifacts/{{project}}/{{run.uid}}
    ```
    
- If the remote service is on an instance of the Iguazio MLOps Platform (**"the platform"**), set the following environment variables as well. Replace the `<...>` placeholders with the details for your specific platform cluster:

    ```ini
    V3IO_USERNAME=<username of a platform user with access to the MLRun service>
    V3IO_API=<API endpoint of the webapi service endpoint; e.g., "https://default-tenant.app.mycluster.iguazio.com:8444">
    V3IO_ACCESS_KEY=<platform access key>
    ```

    You can get the platform access key from the platform dashboard: select the user-profile picture or icon from the top right corner of 
    any page, and select **Access Keys** from the menu. In the **Access Keys** dialog, either copy an existing access key or create a new 
    key and copy it. Alternatively, you can get the access key by checking the value of the `V3IO_ACCESS_KEY` environment variable in a web-
    shell or Jupyter Notebook service.

## Advanced chart configuration

Configurable values are documented in the `values.yaml`, and the `values.yaml` of all sub charts. Override those [in the normal methods](https://helm.sh/docs/chart_template_guide/values_files/).

## Uninstalling the chart

```bash
helm --namespace mlrun uninstall mlrun-kit
```

```{admonition} Terminating pods and hanging resources
This chart generates several persistent volume claims and also provisions an NFS
provisioning server, giving persistency (via PVC) out of the box.
Because of the persistency of PV/PVC resources, after installing this chart, PVs and PVCs are created.
Upon uninstallation, any hanging / terminating pods hold the PVs and PVCs respectively, since those
prevent their safe removal.
Since pods that are stuck in terminating state seem to be a never-ending plague in k8s, note this,
and remember to clean the remaining PVs and PVCs.
```

Handing stuck-at-terminating pods:

```bash
kubectl --namespace mlrun delete pod --force --grace-period=0 <pod-name>
```

Reclaim dangling persistency resources:

```{admonition} WARNING 
**This will result in data loss!**
```

```bash
# To list PVCs
kubectl --namespace mlrun get pvc
...

# To remove a PVC
kubectl --namespace mlrun delete pvc <pvc-name>
...

# To list PVs
kubectl --namespace mlrun get pv
...

# To remove a PV
kubectl --namespace mlrun delete pv <pv-name>

# Remove hostpath(s) used for mlrun (and possibly nfs). Those will be created by default under /tmp, and will contain
# your release name, e.g.:
rm -rf /tmp/mlrun-kit-mlrun-kit-mlrun
...
```

## Upgrading the chart

In order to upgrade to the latest version of the chart, first make sure you have the latest helm repo

```bash
helm repo update
```

Then upgrade the chart:

```bash
helm upgrade --install --reuse-values mlrun-kit v3io-stable/mlrun-kit
```