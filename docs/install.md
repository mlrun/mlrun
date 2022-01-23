# Installation Guide <!-- omit in toc -->

This guide outlines the steps for installing and running MLRun.

- [Install MLRun on a Kubernetes Cluster](#install-mlrun-on-a-kubernetes-cluster)
- [Installing MLRun on a Local Docker Registry](#installing-mlrun-on-a-local-docker-registry)

Once MLRun is installed you can access it remotely from your IDE (PyCharm or VSCode), read [**how to setup your IDE environment**](./howto/remote.md). 


## Install MLRun on a Kubernetes Cluster

### Prerequisites

1. Access to a Kubernetes cluster. You must have administrator permissions in order to install MLRun on your cluster. For local installation on Windows or Mac, we recommend [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. The Kubernetes command-line too (kubectl) compatible with your Kubernetes cluster installed. Refer to the [kubectl installation instructions](https://kubernetes.io/docs/tasks/tools/install-kubectl/) for more information.
3. Helm CLI installed. Refer to the [Helm installation instructions](https://helm.sh/docs/intro/install/) for more information.
4. You must have an accessible docker-registry (such as [Docker Hub](https://hub.docker.com)). The registry's URL and credentials are consumed by the applications via a pre-created secret

> **Note:** These instructions use `mlrun` as the namespace (`-n` parameter). You may want to choose a different namespace in your kubernetes cluster.
<a id="docker-desktop-installation"></a>

### Installing on Docker Desktop

Docker Desktop is available for Mac and Windows. For download information, system requirements, and installation instructions, see:

- [Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/)
- [Install Docker Desktop on Windows](https://docs.docker.com/docker-for-windows/install/). Note that WSL 2 backend was tested, Hyper-V was not tested.

#### Configuring Docker Desktop

Docker Desktop includes a standalone Kubernetes server and client, as well as Docker CLI integration that runs on your machine. The Kubernetes server runs locally within your Docker instance. To enable Kubernetes support and install a standalone instance of Kubernetes running as a Docker container, go to **Preferences** > **Kubernetes** and then click **Enable Kubernetes**. Click **Apply & Restart** to save the settings and then click **Install** to confirm. This instantiates images required to run the Kubernetes server as containers, and installs the `/usr/local/bin/kubectl` command on your machine. For more information, see [the documentation](https://docs.docker.com/desktop/kubernetes/).

We recommend limiting the amount of memory allocated to Kubernetes. If you're using Windows and WSL 2, you can configure global WSL options by placing a `.wslconfig` file into the root directory of your users folder: `C:\Users\<yourUserName>\.wslconfig`. Please keep in mind you may need to run `wsl --shutdown` to shut down the WSL 2 VM and then restart your WSL instance for these changes to take affect.

``` console
[wsl2]
memory=8GB # Limits VM memory in WSL 2 to 8 GB
```

To learn about the various UI options and their usage, see:

- [Docker Desktop for Mac user manual](https://docs.docker.com/docker-for-mac/)
- [Docker Desktop for Windows user manual](https://docs.docker.com/docker-for-windows/)


<a id="installing-the-chart"></a>
### Installing the Chart

Create a namespace for the deployed components:

```bash
kubectl create namespace mlrun
```

Add the `v3io-stable` helm chart repo

```bash
helm repo add v3io-stable https://v3io.github.io/helm-charts/stable
```

Create a secret with your docker-registry named `registry-credentials`:

```bash
kubectl --namespace mlrun create secret docker-registry registry-credentials \
    --docker-server <your-registry-server> \
    --docker-username <your-username> \
    --docker-password <your-password> \
    --docker-email <your-email>
```

where:

- `<your-registry-server>` is your Private Docker Registry FQDN. (https://index.docker.io/v1/ for Docker Hub).
- `<your-username>` is your Docker username.
- `<your-password>` is your Docker password.
- `<your-email>` is your Docker email.

To install the chart with the release name `mlrun-kit` use the following command.
Note the reference to the pre-created `registry-credentials` secret in `global.registry.secretName`:

```bash
helm --namespace mlrun \
    install mlrun-kit \
    --wait \
    --set global.registry.url=<registry-url> \
    --set global.registry.secretName=registry-credentials \
    v3io-stable/mlrun-kit
```

Where `<registry-url` is the registry URL which can be authenticated by the `registry-credentials` secret (e.g., `index.docker.io/<your-username>` for Docker Hub>).


> **Note: Installing on Minikube/VM**
> 
> The Open source MLRun kit uses node ports for simplicity. If your kubernetes cluster is running inside a VM, 
> as is usually the case when using minikube, the kubernetes services exposed over node ports would not be available on 
> your local host interface, but instead, on the virtual machine's interface.
> To accommodate for this, use the `global.externalHostAddress` value on the chart. For example, if you're using 
> the kit inside a minikube cluster (with some non-empty `vm-driver`), pass the VM address in the chart installation 
> command as follows:
>
> ```bash
> helm --namespace mlrun \
>     install my-mlrun \
>     --wait \
>     --set global.registry.url=<registry URL e.g. index.docker.io/iguazio > \
>     --set global.registry.secretName=registry-credentials \
>     --set global.externalHostAddress=$(minikube ip) \
>     v3io-stable/mlrun-kit
> ```
>
> Where `$(minikube ip)` shell command resolving the external node address of the k8s node VM.

### Install Kubeflow

MLRun enables you to run your functions while saving outputs and artifacts in a way that is visible to Kubeflow Pipelines.
If you wish to use this capability you will need to install Kubeflow on your cluster.
Refer to the [**Kubeflow documentation**](https://www.kubeflow.org/docs/started/getting-started/) for more information.

### Usage

Your applications are now available in your local browser:

- Jupyter-notebook - http://localhost:30040
- Nuclio - http://localhost:30050
- MLRun UI - http://localhost:30060
- MLRun API (external) - http://localhost:30070


> **Note:**
>
> The above links assume your Kubernetes cluster is exposed on localhost.
> If that's not the case, the different components will be available on the provided `externalHostAddress`
> - You can change the ports by providing values to the helm install command.
> - You can add and configure a k8s ingress-controller for better security and control over external access.

### Start Working

Open Jupyter Lab on [**jupyter-lab UI**](http://localhost:30040) and run the code in [**docs/quick-start.ipynb**](https://github.com/mlrun/mlrun/blob/master/docs/quick-start.ipynb) notebook.

> **Important:**
>
> Make sure to save your changes in the `data` folder within the Jupyter Lab. The root folder and any other folder will not retain the changes when you restart the Jupyter Lab.

### Configuring Remote Environment

MLRun allows you to use your code on a local machine while running your functions on a remote cluster.

#### Prerequisites

Before you begin, ensure that the following prerequisites are met:

1. Make sure the MLRun version installed with the MLRun Kit is the same as the MLRun version on your remote cluster. If needed, upgrade MLRun either in your local installation or on the remote cluster so they would match.
2. Ensure that you have remote access to your MLRun service (i.e., to the service URL on the remote Kubernetes cluster).

#### Set Environment Variables

Define your MLRun configuration. As a minimum requirement:

1. Set `MLRUN_DBPATH` to the URL of the remote MLRun database/API service; replace the `<...>` placeholders to identify your remote target:

    ```ini
    MLRUN_DBPATH=<API endpoint of the MLRun APIs service endpoint; e.g., "https://mlrun-api.default-tenant.app.mycluster.iguazio.com">
    ```

2. In order to store the artifacts on the remote server, you need to set the `MLRUN_ARTIFACT_PATH` to the desired root folder of your artifact. You can use `{{project}}` to include the project name in the path `{{run.uid}}` to include the specific run uid in the artifact path. For example:

    ```ini
    MLRUN_ARTIFACT_PATH=/User/artifacts/{{project}}
    ```

3. If the remote service is on an instance of the Iguazio MLOps Platform (**"the platform"**), set the following environment variables as well; replace the `<...>` placeholders with the information for your specific platform cluster:

    ```ini
    V3IO_USERNAME=<username of a platform user with access to the MLRun service>
    V3IO_API=<API endpoint of the webapi service endpoint; e.g., "https://default-tenant.app.mycluster.iguazio.com:8444">
    V3IO_ACCESS_KEY=<platform access key>
    ```

    You can get the platform access key from the platform dashboard: select the user-profile picture or icon from the top right corner of any page, and select **Access Keys** from the menu. In the **Access Keys** window, either copy an existing access key or create a new key and copy it. Alternatively, you can get the access key by checking the value of the `V3IO_ACCESS_KEY` environment variable in a web-shell or Jupyter Notebook service.

### Advanced Chart Configuration

Configurable values are documented in the `values.yaml`, and the `values.yaml` of all sub charts. Override those [in the normal methods](https://helm.sh/docs/chart_template_guide/values_files/).

### Uninstalling the Chart

```bash
helm --namespace mlrun uninstall mlrun-kit
```

> **Note on terminating pods and hanging resources:**
> 
> It is important to note that this chart generates several persistent volume claims and also provisions an NFS
provisioning server, to provide the user with persistency (via PVC) out of the box.
> Because of the persistency of PV/PVC resources, after installing this chart, PVs and PVCs will be created,
And upon uninstallation, any hanging / terminating pods will hold the PVCs and PVs respectively, as those
Prevent their safe removal.
> Because pods stuck in terminating state seem to be a never-ending plague in k8s, please note this,
And don't forget to clean the remaining PVCs and PVCs

Handing stuck-at-terminating pods:

```bash
kubectl --namespace mlrun delete pod --force --grace-period=0 <pod-name>
```

Reclaim dangling persistency resources:

| WARNING: This will result in data loss! |
| --- |

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


<a id="local-docker"></a>
## Installing MLRun on a Local Docker Registry

To use MLRun with your local Docker registry, run the MLRun API service, dashboard, and example Jupyter server by using the following script.

> **Note:**
> - Using Docker is limited to local runtimes.
> - By default, the MLRun API service will run inside the Jupyter server, set the MLRUN_DBPATH env var in Jupyter to point to an alternative service address.
> - The artifacts and DB will be stored under **/home/jovyan/data**, use docker -v option to persist the content on the host (e.g. `-v $(SHARED_DIR}:/home/jovyan/data`)
> - If Docker is running on Windows with WSL 2, you must create SHARED_DIR before running these commadns. Provide the full path when executing  (e.g. `mkdir /mnt/c/mlrun-data`  `SHARED_DIR=/mnt/c/mlrun-data`)

```sh
SHARED_DIR=~/mlrun-data

docker pull mlrun/jupyter:0.9.2
docker pull mlrun/mlrun-ui:0.9.2

docker network create mlrun-network
docker run -it -p 8080:8080 -p 30040:8888 --rm -d --network mlrun-network --name jupyter -v ${SHARED_DIR}:/home/jovyan/data mlrun/jupyter:0.9.2
docker run -it -p 30050:80 --rm -d --network mlrun-network --name mlrun-ui -e MLRUN_API_PROXY_URL=http://jupyter:8080 mlrun/mlrun-ui:0.9.2
```

When the execution completes &mdash;

- Open Jupyter Lab on port 30040 and run the code in the [**mlrun_basics.ipynb**](https://github.com/mlrun/mlrun/blob/master/examples/mlrun_basics.ipynb) notebook.
- Use the MLRun dashboard on port 30050.
