(install-aws)=
# Install MLRun CE on AWS

```{admonition} Note
These instructions install the community edition (CE). 
```

**In this section**
- [Prerequisites](#prerequisites)
- [Community Edition flavors](#community-edition-flavors)
- [Installing the chart](#installing-the-chart)
- [Configuring the online features store](#configuring-the-online-feature-store)
- [Usage](#usage)
- [Start working](#start-working)
- [Configuring the remote environment](#configuring-the-remote-environment)
- [Advanced chart configuration](#advanced-chart-configuration)
- [Storage resources](#storage-resources)
- [Uninstalling the chart](#uninstalling-the-chart)
- [Upgrading the chart](#upgrading-the-chart)
- [Storing artifacts in AWS S3 storage](#storing-artifacts-in-aws-s3-storage)

## Prerequisites

- A registered domain name allowing wildcards with a dummy CNAME record (will be filled later with the AWS Load Balancer CNAME)
- AWS CLI is installed and configured. See [Installing or updating to the latest version of the AWS CLI - AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- eksctl is installed and configured. See [What is Amazon EMR on EKS? - Amazon EMR](https://docs.aws.amazon.com/emr/latest/EMR-on-EKS-DevelopmentGuide/emr-eks.html),  [Installation - eksctl](https://eksctl.io/installation/) 
- kubectl is installed. See Set up kubectl and eksctl - Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html)
- Helm is installed. See [Deploy applications with Helm on Amazon EKS - Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/helm.html)
- A bash shell to run the commands        

```{admonition} Important
Restart your MAC after following the prerequisites steps to make the changes take effect.
```

``` {admonition} Note
The MLRun Community Edition resources are configured initially with the default cluster/namespace resource limits. You can modify the resources from outside if needed.
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
* MinIO - https://github.com/minio/minio/tree/master/helm/minio
* Spark Operator - https://github.com/GoogleCloudPlatform/spark-on-k8s-operator
* Pipelines - https://github.com/kubeflow/pipelines
* Prometheus stack - https://github.com/prometheus-community/helm-charts
  - Prometheus
  - Grafana


## Configuring the online feature store
The MLRun Community Edition supports the online feature store. To enable it, you need to first deploy a Redis service that is accessible to your MLRun CE cluster.
To deploy a Redis service, refer to the [Redis documentation](https://redis.io/docs/getting-started/).

When you have a Redis service deployed, you can configure MLRun CE to use it by adding the following helm value configuration to your helm install command:
```bash
--set mlrun.api.extraEnvKeyValue.MLRUN_REDIS__URL=<redis-address>
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


## Installation
1. [Optional] Create or import a certificate to AWS Certificate Manager for the relevant domain including wildcards **\*.SYSTEM_NAME.example.com** by one of:
    - Create a certificate [AWS Certificate Manager public certificates - AWS Certificate Manager](https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-request-public.html)
	- Import an existing certificate [Import a certificate - AWS Certificate Manager] (https://docs.aws.amazon.com/acm/latest/userguide/import-certificate-api-cli.html)</br>
    Note the ARN of the certificate
2. Export the following env variables, fill in the relevant <SYSTEM_NAME> and <DOMAIN_NAME>:
   - export SYSTEM_NAME="<SYSTEM_NAME>"
   - export DOMAIN_NAME="<DOMAIN_NAME>"
2.Export a comma-delimited list of CIDR ranges that will be able to access the MLRun services via the AWS ALB:
   - export INBOUND_CIDRS="<CIDR_RANGE>[,<CIDR_RANGE>].."</br>
   Ensure the CIDR_RANGE is correctly formatted, including the subnet mask (e.g. 192.168.1.0/24).
2. Export the remaining derived values:
   - export USER_NAME=$(aws iam get-user --query 'User.UserName' --output text)
   - export ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
   - export REGION=$(aws configure get region)
   - export SYSTEM_FQDN="${SYSTEM_NAME}.${DOMAIN_NAME}"
   - export BUCKET_NAME="${SYSTEM_NAME}-${ACCOUNT_ID}-bucket"
   - export ECR_REPO_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${SYSTEM_NAME}"
2. Create the EKS cluster
    1. Download the following EKS config file template:

Open cluster.yaml.template

cluster.yaml.template
03 Dec 2024, 02:06 PM
Create an EKS cluster.yaml config file from the downloaded template and the env variables



envsubst < cluster.yaml.template > cluster.yaml
The minimal instance size required for MLRun to operate is m5.xlarge, you can increase the instance numbers and sizes in the cluster.yaml per your requirements.

Create an EKS cluster using the cluster.yaml conifg file:



eksctl create cluster -f cluster.yaml
The installation will also create the mlrun namespace and add IAM roles, policies and service accounts for EBS and S3 access.

Configure EBS as the default Storage Class


kubectl patch storageclass gp2 -p '{"metadata": {"annotations": {"storageclass.kubernetes.io/is-default-class": "true"}}}'
Create an S3 bucket to store MLRun artifacts


aws s3 mb s3://"${BUCKET_NAME}" --region "${REGION}"
Get the cluster’s VPC ID


export VPC_ID=$(aws eks describe-cluster \
          --name "${SYSTEM_NAME}" \
          --query "cluster.resourcesVpcConfig.vpcId" \
          --output text)
echo VPC_ID=${VPC_ID}
Create a Gateway Endpoint to access the bucket directly from the VPC 
Get the route-table IDs of the VPC



TABLES_ARRAY=($(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=${VPC_ID}" --query 'RouteTables[*].RouteTableId' --output text))
echo TABLES_ARRAY=${TABLES_ARRAY[@]}
Create the Endpoint



aws ec2 create-vpc-endpoint \
  --vpc-id ${VPC_ID} \
  --service-name com.amazonaws.${REGION}.s3 \
  --vpc-endpoint-type Gateway \
  --region ${REGION} \
  --route-table-ids ${TABLES_ARRAY[@]}
Install the AWS Load Balancer Controller


helm repo add eks https://aws.github.io/eks-charts
helm repo update
helm install aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system --set clusterName="${SYSTEM_NAME}" \
    --set serviceAccount.create=false \
    --set serviceAccount.name=aws-load-balancer-controller-sa \
    --set vpcId=${VPC_ID}
Install MLRun CE with aws_values.yaml file into the mlrun namespace
Download the following aws_values file template:

Open aws_values.yaml.template

aws_values.yaml.template
03 Dec 2024, 02:01 PM
Create the aws_values.yaml config file from the downloaded template and the env variables



envsubst < aws_values.yaml.template > aws_values.yaml
Add MLRun CE helm repo



helm repo add mlrun-ce https://mlrun.github.io/ce
helm repo update
If you do not have a certificate, Run the following to install the MLRun CE helm chart using the values file



helm install --wait --dependency-update --namespace mlrun -f aws_values.yaml mlrun-ce mlrun-ce/mlrun-ce --version 0.7.0
Alternatively, if you have a certificate, add the CERTIFICATE_ARN to the install command



helm install --wait --dependency-update --namespace mlrun -f aws_values.yaml --set global.domainNameCertificate="<CERTIFICATE ARN>" mlrun-ce mlrun-ce/mlrun-ce --version 0.7.0
Get the AWS Load Balancer CNAME and set is as a value for your DNS record
Configure the CNAME in your domain, pointing *.<system_name>.<domain> to the Load Balancer URL:



kubectl -n mlrun get ingress mlrun-ce-ingress -o custom-columns=":status.loadBalancer.ingress[0].hostname" --no-headers
[Optional] Add access to the EKS cluster to additional users
To allow access to the EKS API for additional users please refer to the following AWS documentation Grant IAM users and roles access to Kubernetes APIs - Amazon EKS 

[Optional] Grant access to the S3 bucket to additional users
To allow access to the S3 bucket for additional users please refer to the following AWS walkthrough Example 1: Bucket owner granting its users bucket permissions - Amazon Simple Storage Service 

 

Uninstalling the cluster and deleting the resources
Export the following env variables, Fill the desired <SYSTEM_NAME>, <DOMAIN_NAME>:


export SYSTEM_NAME="<SYSTEM_NAME>"
export DOMAIN_NAME="<DOMAIN_NAME>"


export USER_NAME=$(aws iam get-user --query 'User.UserName' --output text)
export ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
export REGION=$(aws configure get region)
export SYSTEM_FQDN="${SYSTEM_NAME}.${DOMAIN_NAME}"
export BUCKET_NAME="${SYSTEM_NAME}-${ACCOUNT_ID}-bucket"
export ECR_REPO_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${SYSTEM_NAME}"
Get the cluster’s VPC ID


export VPC_ID=$(aws eks describe-cluster \
          --name "${SYSTEM_NAME}" \
          --query "cluster.resourcesVpcConfig.vpcId" \
          --output text)
echo VPC_ID=${VPC_ID}
Delete the S3 gateway endpoint


ENDPOINT_IDS=$(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=${VPC_ID}" --query 'VpcEndpoints[*].VpcEndpointId' --output text)
for ENDPOINT_ID in ${ENDPOINT_IDS}; do
  echo "Deleting VPC Endpoint: ${ENDPOINT_ID}"
  aws ec2 delete-vpc-endpoints --vpc-endpoint-ids "${ENDPOINT_ID}"
done
Delete the S3 bucket


aws s3 rm s3://${BUCKET_NAME} --recursive
aws s3 rb s3://${BUCKET_NAME} --force
Delete the ECR repositories


# Get all the repositories names
REPO_NAMES=$(aws ecr describe-repositories --region "${REGION}" --query 'repositories[?starts_with(repositoryName, `'${SYSTEM_NAME}'`)].repositoryName' --output text)
# Loop through each repository
for REPO_NAME in ${REPO_NAMES}; do
  # Get all image tags in the repository
  IMAGE_TAGS=$(aws ecr list-images --repository-name "${REPO_NAME}" --region "${REGION}" --query 'imageIds[].imageTag' --output text)
  # Get all image digests in the repository
  IMAGE_DIGESTS=$(aws ecr list-images --repository-name "${REPO_NAME}" --region "${REGION}" --query 'imageIds[].imageDigest' --output text)
  # Delete images by tag
  for TAG in ${IMAGE_TAGS}; do
    if [ -n "${TAG}" ]; then
      echo "Deleting image ${REPO_NAME}:${TAG}"
      aws ecr batch-delete-image --repository-name "${REPO_NAME}" --region "${REGION}" --image-ids imageTag="${TAG}"
    fi
  done
  # Delete images by digest
  for DIGEST in ${IMAGE_DIGESTS}; do
    if [ -n "${DIGEST}" ]; then
      echo "Deleting image ${REPO_NAME}:${DIGEST}"
      aws ecr batch-delete-image --repository-name "${REPO_NAME}" --region "${REGION}" --image-ids imageDigest="${DIGEST}"
    fi
  done
  # Delete the repository itself
  aws ecr delete-repository --repository-name "${REPO_NAME}" --region "${REGION}" --force
done
Delete the EKS cluster


eksctl delete cluster --name "${SYSTEM_NAME}"
Delete the EBS volume leftovers
Get the relevant volume IDs


VOLUME_IDS=$(aws ec2 describe-volumes --region "${REGION}" \
  --query "Volumes[?not_null(Tags[?Key=='Name']|[0].Value) && starts_with(Tags[?Key=='Name']|[0].Value, \`${SYSTEM_NAME}\`)].VolumeId" \
  --output text)
echo VOLUME_IDS=${VOLUME_IDS}
Delete the volumes


for VOLUME_ID in ${VOLUME_IDS}; do
    aws ec2 delete-volume --volume-id ${VOLUME_ID} --region ${REGION}
done
























OLD OLD OLD



## Configuring the remote environment

You can use your code on a local machine while running your functions on a remote cluster. Refer to [Set up your environment](https://docs.mlrun.org/en/latest/install/remote.html) for more information.

## Advanced chart configuration

Configurable values are documented in the `values.yaml`, and the `values.yaml` of all sub charts. Override those [in the normal methods](https://helm.sh/docs/chart_template_guide/values_files/).

### Opt out of components
The chart installs many components. You may not need them all in your deployment depending on your use cases.
To opt out of some of the components, use the following helm values:
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
running as a Docker container, go to **Preferences** > **Kubernetes** and then press **Enable Kubernetes**. Press **Apply & Restart** to 
save the settings and then press **Install** to confirm. This instantiates the images that are required to run the Kubernetes server as 
containers, and installs the `/usr/local/bin/kubectl` command on your machine. For more information, see [the Kubernetes documentation](https://docs.docker.com/desktop/kubernetes/).

It's recommended to limit the amount of memory allocated to Kubernetes. If you're using Windows and WSL 2, you can configure global WSL options by placing a `.wslconfig` file into the root directory of 
your users folder: `C:\Users\<yourUserName>\.wslconfig`. Keep in mind that you might need to run `wsl --shutdown` to shut down the WSL 2 VM and then restart your WSL instance for these changes to take effect.

``` console
[wsl2]
memory=8GB # Limits VM memory in WSL 2 to 8 GB
```

To learn about the various UI options and their usage, see:

- [Docker Desktop for Mac user manual](https://docs.docker.com/docker-for-mac/)
- [Docker Desktop for Windows user manual](https://docs.docker.com/docker-for-windows/)

## Storage resources

When installing the MLRun Community Edition, several storage resources are created:

- **PVs via default configured storage class**: Holds the file system of the stacks pods, including the MySQL database of MLRun, MinIO for artifacts and Pipelines Storage and more. 
These are not deleted when the stack is uninstalled, which allows upgrading without losing data.
- **Container Images in the configured docker-registry**: When building and deploying MLRun and Nuclio functions via the MLRun Community Edition, the function images are 
stored in the given configured docker registry. These images persist in the docker registry and are not deleted.



### Notes on dangling resources
- The created CRDs are not deleted by default and should be manually cleaned up. 
- The created PVs and PVCs are not deleted by default and should be manually cleaned up. 
- As stated above, the images in the docker registry are not deleted either and should be cleaned up manually.
- If you installed the chart in its own namespace, it's also possible to delete the entire namespace to clean up all resources (apart from the docker registry images).

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

# To remove a PVC
$ kubectl --namespace mlrun delete pvc <pv-name>
...
```




### Setting up S3 credentials and endpoint

Set up the following project-secrets (refer to [**Data stores**](../store/datastore.md) and [**Project secrets**](../secrets.md#mlrun-managed-secrets)) 
for any project used:

* `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` &mdash; S3 credentials
* `S3_ENDPOINT_URL` &mdash; the AWS S3 endpoint to use, depending on the region. For example: 
    ``` console
    S3_ENDPOINT_URL = https://s3.us-east-2.amazonaws.com/
    ```

### Disabling auto-mount

Before running any MLRun job that writes to S3 bucket, make sure auto-mount is disabled for it, since by default
auto-mount adds S3 configurations that point at the MinIO service (refer to 
[**Function storage**](../runtimes/function-storage.md) for more details on auto-mount). This can be done in one
of following ways:

* Set the client-side MLRun configuration to disable auto-mount. This disables auto-mount for any function run
  after this command:
    ```python
    from mlrun.config import config as mlconf

    mlconf.storage.auto_mount_type = "none"
    ```
* If running MLRun from an IDE, the configuration can be overridden using an environment variable. Set the following
  environment variable for your IDE environment:
    ```python
    MLRUN_STORAGE__AUTO_MOUNT_TYPE = "none"
    ```
* Disable auto-mount for a specific function. This must be done before running the function for the first time:
    ```python
    function.spec.disable_auto_mount = True
    ```

### Changing the artifact path

The artifact path needs to be modified since the bucket name is set to `mlrun` by default. It is recommended to keep 
the same path structure as the default, while modifying the bucket name. For example:
```text
s3://<bucket name>/projects/{{run.project}}/artifacts
```

The artifact path can be set in several ways, refer to [**Artifact path**](../store/artifacts.md#artifact-path) 
for more details.

```{admonition} Note
If your values have fixed mlrun service versions (e.g.: mlrun:1.5.0) then you might want to remove it from the values file to allow newer chart defaults to kick in.
```
