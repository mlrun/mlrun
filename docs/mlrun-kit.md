# MLRun Kit: Open Source Bundle

*The fastest way to develop and deploy your AI application today.*

## Overview

**MLRun Kit** is the first end-to-end open source MLOps solution to manage and automate your entire analytics and machine learning lifecycle, from data ingestion through model development and full pipeline deployment.

### Benefits

With the MLRun Kit you can:

- Develop your training pipeline on any framework locally and/or on a cluster.
- Leverage the power of the open source function marketplace to focus on your research.
- Deploy your pipeline at scale in a single click.
- Monitor your model performance and automate your actions.

### Components

The MLRun Kit includes the following components:

- Project lifecycle management: experiment management and tracking of jobs, functions and artifacts.
- Scalable functions: turn code to scalable microservices in a single command.
- Managed Pipelines: deploy, run and monitor your machine learning execution plan.

## Installation

This Helm charts bundles open source software stack for advanced ML operations

### Chart Details

The MLRun Kit chart includes the following stack:

- MLRun - <https://github.com/mlrun/mlrun>
- Nuclio - <https://github.com/nuclio/nuclio>
- Jupyter - <https://github.com/jupyterlab/jupyterlab> (+MLRun integrated)
- NFS - <https://github.com/kubernetes-retired/external-storage/tree/master/nfs>
- MPI Operator - <https://github.com/kubeflow/mpi-operator>

### Installing the Chart

Create a namespace for the deployed components:

```bash
kubectl create namespace mlrun
```

Add the `v3io-stable` helm chart repo

```bash
helm repo add v3io-stable https://v3io.github.io/helm-charts/stable
```

To work with the MLRun stack, you must an accessible docker-registry. The registry's URL and credentials
are consumed by the applications via a pre-created secret

To create a secret with your docker-registry details:

```bash
kubectl --namespace mlrun create secret docker-registry registry-credentials \
    --docker-username <registry-username> \
    --docker-password <login-password> \
    --docker-server <server URL, e.g. https://index.docker.io/v1/ > \
    --docker-email <user-email>
```

To install the chart with the release name `my-mlrun` use the following command.
Note the reference to the pre-created `registry-credentials` secret in `global.registry.secretName`, and a `global.registry.url` with an appropriate registry URL which can be authenticated by this secret:

```bash
helm --namespace mlrun \
    install my-mlrun \
    --render-subchart-notes \
    --set global.registry.url=<registry URL e.g. index.docker.io/iguazio > \
    --set global.registry.secretName=registry-credentials \
    v3io-stable/mlrun-kit
```

### Install Kubeflow

MLRun enables you to run your functions while saving outputs and artifacts in a way that is visible to Kubeflow Pipelines.
If you wish to use this capability you will need to install Kubeflow on your cluster.
Refer to the [**Kubeflow documentation**](https://www.kubeflow.org/docs/started/getting-started/) for more information.

### Usage

Your applications are now available in your local browser:

- Jupyter-Lab - <http://localhost:30040>
- MLRun - <http://locahost:30050>
- Nuclio - <http://localhost:30060>

### Start Working

Open Jupyter Lab on [**jupyter-lab UI**](http://localhost:30040) and run the code in [**examples/mlrun_basics.ipynb**](https://github.com/mlrun/mlrun/blob/master/examples/mlrun_basics.ipynb) notebook.

> **Note:**
>
> - You can change the ports by providing values to the helm install command.
> - You can add and configure a k8s ingress-controller for better security and control over external access.

### Configuring Remote Environment

MLRun allows you to use your code on a local machine while running your functions on a remote cluster.

#### Prerequisites

Before you begin, ensure that the following prerequisites are met:

1. Make sure the MLRun version installed with the MLRun Kit is the same as the MLRun version on your remote cluster. If needed, upgrade MLRun either in your local installation or on the remote cluster so they would match.
2. Ensure that you have remote access to your MLRun service (i.e., to the service URL on the remote Kubernetes cluster).

#### Set Environment Variables

Define your MLRun configuration. As a minimum requirement &mdash;

1. Set `MLRUN_DBPATH` to the URL of the remote MLRun database/API service; replace the `<...>` placeholders to identify your remote target:

    ```ini
    MLRUN_DBPATH=<API endpoint of the MLRun APIs service engpoint; e.g., "https://mlrun-api.default-tenant.app.mycluster.iguazio.com">
    ```

2. In order to store the artifacts on the remote server, you need to set the `MLRUN_ARTIFACT_PATH` to the desired root folder of your artifact. You can use `{{run.project}}` to include the project name in the path `{{run.uid}}` to include the specific run uid in the artifact path. For example:

    ```ini
    MLRUN_ARTIFACT_PATH=/User/artifacts/{{run.project}}
    ```

3. If the remote service is on an instance of the Iguazio Data Science Platform (**"the platform"**), set the following environment variables as well; replace the `<...>` placeholders with the information for your specific platform cluster:

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
helm --namespace mlrun uninstall my-mlrun
```

#### Note on terminating pods and hanging resources

It is important to note that this chart generates several persistent volume claims and also provisions an NFS
provisioning server, to provide the user with persistency (via PVC) out of the box.
Because of the persistency of PV/PVC resources, after installing this chart, PVs and PVCs will be created,
And upon uninstallation, any hanging / terminating pods will hold the PVCs and PVs respectively, as those
Prevent their safe removal.
Because pods stuck in terminating state seem to be a never-ending plague in k8s, please note this,
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
rm -rf /tmp/my-mlrun-mlrun-kit-mlrun
...
```
