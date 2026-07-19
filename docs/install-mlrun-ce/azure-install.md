(azure-install)=
# Install MLRun CE on Azure

These instructions explain how to install MLRun CE on your Azure Kubernetes Service (AKS) cluster.

```{admonition} Note
These instructions install the community edition {{ ceversion }}, which currently includes the features in MLRun v{{ version }}.
```

**In this section**
- [Prerequisites](#prerequisites)
- [Community Edition services](#community-edition-services)
- [Prepare values files](#prepare-values-files)
- [Create the container registry secret](#create-the-container-registry-secret)
- [Installation](#installation)
- [Usage](#usage)
- [Optional: Ingress configuration](#optional-ingress-configuration)
- [Uninstalling the chart](#uninstalling-the-chart)

## Prerequisites

- An AKS cluster (Kubernetes >=1.34) with a default StorageClass configured
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) is installed and logged in (`az login`)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) is configured for your AKS cluster. See [Connect to an Azure Kubernetes Service (AKS) cluster using kubectl](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-cli)
- [Helm](https://helm.sh/docs/intro/install/) version >=4.1 CLI is installed
- An accessible Docker Registry (such as [Docker Hub](https://hub.docker.com) or [Azure Container Registry (ACR)](https://learn.microsoft.com/en-us/azure/container-registry/)). The Registry's URL and credentials are consumed by the applications via a pre-created secret.
- Review the [MLRun CE installation notes](./mlrun-ce-installation-notes.md) for any additional installation steps you may need to consider.
- A bash shell to run the commands
- [Optional] If you want to store MLRun artifacts in Azure Blob Storage instead of the bundled SeaweedFS, provision an Azure Storage account and container. See [Using Azure Blob Storage for MLRun artifacts](./mlrun-ce-installation-notes.md#using-azure-blob-storage-for-mlrun-artifacts).

```{admonition} Note
The MLRun Community Edition resources are configured initially with the default cluster/namespace resource limits. You can modify the resources from outside if needed.
```

## Community Edition services

The MLRun CE (Community Edition) includes the following services:
- MLRun - https://github.com/mlrun/mlrun
  - MLRun API
  - MLRun UI
  - MLRun DB (MySQL)
- Nuclio - https://github.com/nuclio/nuclio
- Jupyter - https://github.com/jupyter/notebook (+MLRun integrated)
- SeaweedFS - https://github.com/seaweedfs/seaweedfs (S3-compatible storage)
- Kafka
- TimescaleDB
- Prometheus stack - https://github.com/prometheus-community/helm-charts
  - Prometheus
  - Grafana

## Prepare values files

Create `azure-values.yaml` with your container registry and storage settings.

1. Download the {download}`azure values file template <./azure-values.yaml.template>`.
2. Copy and edit `azure-values.yaml` — fill in your registry URL, secret name, and storage settings:

```yaml
storage:
  mode: "azure-blob"
  azure:
    containerName: "data"
    accountName: "<name of the storage account>"
    connectionString: "DefaultEndpointsProtocol=https;AccountName=<name of the storage>;AccountKey=<account key>;EndpointSuffix=core.windows.net"

global:
  registry:
    url: "<url of the registry>"
    secretName: "acr-registry-credentials"
```

### Storage options

`azure-values.yaml` controls where MLRun and Jupyter store artifacts:

**Option A — Azure Blob Storage**

```yaml
storage:
  mode: "azure-blob"
  azure:
    containerName: "data"
    accountName: "<name of the storage account>"
    connectionString: "<connection-string>"
```

See also [Using Azure Blob Storage for MLRun artifacts](./mlrun-ce-installation-notes.md#using-azure-blob-storage-for-mlrun-artifacts).

**Option B — Bundled SeaweedFS (default)**

Set `storage.mode` to `"local"` to use the in-cluster SeaweedFS S3-compatible service. Kubeflow Pipelines always uses SeaweedFS regardless of this setting.

```yaml
storage:
  mode: "local"
  local:
    accessKey: "seaweed"
    secretKey: "seaweed123"
    bucket: "mlrun"
```

## Create the container registry secret

Create a namespace and a Kubernetes secret with your registry credentials.

```bash
kubectl create namespace mlrun
```

**Using Azure Container Registry (ACR):**

```bash
export ACR_NAME="<acr-name>"
export ACR_TOKEN=$(az acr login --name "${ACR_NAME}" --expose-token --output tsv --query accessToken)

kubectl --namespace mlrun create secret docker-registry acr-registry-credentials \
    --docker-server="<url of the registry>" \
    --docker-username="00000000-0000-0000-0000-000000000000" \
    --docker-password="${ACR_TOKEN}" \
    --docker-email="not-used@example.com"
```

The secret name must match `global.registry.secretName` in `azure-values.yaml` (default: `acr-registry-credentials`).

**Using Docker Hub or another private Docker registry:**

```bash
kubectl --namespace mlrun create secret docker-registry acr-registry-credentials \
    --docker-server="<registry-server>" \
    --docker-username="<registry-username>" \
    --docker-password="<registry-password>" \
    --docker-email="<user-email>"
```

## Installation

1. Add the MLRun CE Helm repo:

```bash
helm repo add mlrun-ce https://mlrun.github.io/ce
helm repo update
```

2. Install MLRun CE using `azure-values.yaml`:

```bash
helm install mlrun-ce mlrun-ce/mlrun-ce \
  --namespace mlrun \
  --wait \
  --timeout 2000s \
  -f azure-values.yaml
```

3. Verify the installation:

```bash
kubectl -n mlrun get pods
```

## Usage

By default, MLRun CE exposes services via NodePort. Your applications are available at:

- Jupyter Notebook - `http://<node-ip>:30040`
- Nuclio - `http://<node-ip>:30050`
- MLRun UI - `http://<node-ip>:30060`
- MLRun API (external) - `http://<node-ip>:30070`
- SeaweedFS Admin - `http://<node-ip>:30093`
- Pipeline UI - `http://<node-ip>:30100`
- Grafana UI - `http://<node-ip>:30010`
- Prometheus UI - `http://<node-ip>:30020`

Replace `<node-ip>` with your AKS node external IP or `global.externalHostAddress` if configured.

Open the Jupyter notebook on [**jupyter-notebook UI**](http://localhost:30040) and run the code in the
[**examples/mlrun_basics.ipynb**](https://github.com/mlrun/mlrun/blob/master/examples/mlrun_basics.ipynb) notebook.

## Optional: Ingress configuration

If you prefer ingress-based access instead of NodePort, optionally add `values-ingress-override-azure.yaml` to the install command. This switches exposed services to ClusterIP and enables ingress for MLRun UI, MLRun API, Nuclio, Jupyter, SeaweedFS Admin, Grafana, and Prometheus.

1. Download the {download}`ingress override values file template <./values-ingress-override-azure.yaml.template>`.
2. Generate the override file. Export your FQDN and ingress class, then run:

```bash
export SYSTEM_FQDN="<system-fqdn>"
export INGRESS_CLASS_NAME="traefik"

envsubst < values-ingress-override-azure.yaml.template > values-ingress-override-azure.yaml
```

3. Install (or upgrade) with both values files:

```bash
helm install mlrun-ce mlrun-ce/mlrun-ce \
  --namespace mlrun \
  --wait \
  --timeout 2000s \
  -f azure-values.yaml \
  -f values-ingress-override-azure.yaml
```

4. Configure DNS records for your ingress hostnames, pointing each host to your ingress controller's external IP or load balancer:

   - `mlrun.${SYSTEM_FQDN}`
   - `mlrun-api.${SYSTEM_FQDN}`
   - `nuclio.${SYSTEM_FQDN}`
   - `jupyter.${SYSTEM_FQDN}`
   - `seaweedfs.${SYSTEM_FQDN}`
   - `grafana.${SYSTEM_FQDN}`
   - `prometheus.${SYSTEM_FQDN}`

With ingress enabled, your applications are available at:

- MLRun UI - `https://mlrun.${SYSTEM_FQDN}`
- MLRun API - `https://mlrun-api.${SYSTEM_FQDN}`
- Nuclio - `https://nuclio.${SYSTEM_FQDN}`
- Jupyter Notebook - `https://jupyter.${SYSTEM_FQDN}`
- SeaweedFS Admin - `https://seaweedfs.${SYSTEM_FQDN}`
- Grafana - `https://grafana.${SYSTEM_FQDN}`
- Prometheus - `https://prometheus.${SYSTEM_FQDN}`

## Uninstalling the chart

```bash
helm --namespace mlrun uninstall mlrun-ce
```

See the [Kubernetes uninstall notes](./kubernetes-install.md#uninstalling-the-chart) for guidance on cleaning up persistent volumes and other dangling resources.
