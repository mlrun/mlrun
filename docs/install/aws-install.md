(install-on-aws)=
# Install MLRun on AWS

For AWS users, the easiest way to install MLRun is to use a native AWS deployment. This option deploys MLRun on an AWS EKS service.

```{admonition} Note
These instructions install the community edition, which currently includes MLRun {{ ceversion }}. See the {{ '[release documentation](https://{})'.format(releasedocumentation) }}.
```
## Prerequisites
- A registered domain name allowing wildcards with a dummy CNAME record (will be filled later with the AWS Load Balancer CNAME)
- [AWS CLI installed and configured](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)  
- [eksctl installed and configured](https://docs.aws.amazon.com/emr/latest/EMR-on-EKS-DevelopmentGuide/setting-up-eksctl.html) 
- [kubectl installed](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html) 
- [helm installed](https://docs.aws.amazon.com/eks/latest/userguide/helm.html) 
- A bash shell to run the commands
```{admonition} Note
Make sure to restart your MAC after following the prerequisites steps to take the changes into effect.
```
        

## IAM Requirements
Verify that your AWS account has the following IAM policies:


````{toggle} show the IAM policies file
   ```{literalinclude} ./iam-policies.yaml
   :language: yaml
   ```
````

## Installation

### [Optional] Get/create an AWS Certificate Manager
Create or import a certificate to AWS Certificate Manager for the relevant domain including wildcards *.cluster_name.example.com
1. Do one of:
   - [Create a certificate](https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-request-public.html)
   - [Import an existing one](https://docs.aws.amazon.com/acm/latest/userguide/import-certificate-api-cli.html)
2. Note the ARN of the certificate.

### Configure environment variables
Export the following env variables, fill in the desired <CLUSTER_NAME> and the <DOMAIN_NAME>:
```python
export CLUSTER_NAME="<CLUSTER_NAME>"
export DOMAIN_NAME="<DOMAIN_NAME>"
```

### Export CIDR ranges  
Export a comma-delimited list of CIDR ranges that will be able to access the MLRun services via the AWS ALB:

```python
export INBOUND_CIDRS="<CIDR_RANGE>[,<CIDR_RANGE>].."
```

### Export the remaining derived values
```python
export USER_NAME=$(aws iam get-user --query 'User.UserName' --output text)
export ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
export REGION=$(aws configure get region)
export CLUSTER_FQDN="${CLUSTER_NAME}.${DOMAIN_NAME}"
export BUCKET_NAME="${CLUSTER_NAME}-${ACCOUNT_ID}-bucket"
export ECR_REPO_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${CLUSTER_NAME}"
```
### Create the EKS cluster

{Download}`[Download the EKS cluster.yaml config file]<./cluster.yaml.template>`

Create an EKS cluster.yaml config file from the downloaded template and the env variables.
```python
envsubst < cluster.yaml.template > cluster.yaml
```
The minimal instance size required for MLRun to operate is m5.xlarge, you can increase the instance numbers and sizes in the cluster.yaml per your requirements.

Create an EKS cluster using the cluster.yaml conifg file:

```python
eksctl create cluster -f cluster.yaml
```
The installation will also create the mlrun namespace and add IAM roles, policies and service accounts for EBS, EFS and S3 access.

### Configure EBS as the default Storage Class
```python
kubectl patch storageclass gp2 -p '{"metadata": {"annotations": {"storageclass.kubernetes.io/is-default-class": "true"}}}'
```
### Create an EFS filesystem and Storage Class with mount targets in the cluster VPC subnets
```python
export FILE_SYSTEM_ID=$(aws efs create-file-system \
          --region "${REGION}" \
          --performance-mode generalPurpose \
          --query 'FileSystemId' \
          --tags Key=Owner,Value="${USER_NAME}" Key=Cluster,Value="${CLUSTER_NAME}" \
          --output text)
echo FILE_SYSTEM_ID=${FILE_SYSTEM_ID}
```
Get the cluster’s VPC ID
```python
export VPC_ID=$(aws eks describe-cluster \
          --name "${CLUSTER_NAME}" \
          --query "cluster.resourcesVpcConfig.vpcId" \
          --output text)
echo VPC_ID=${VPC_ID}
```
Get the VPC CIDR range
```python
export CIDR_RANGE=$(aws ec2 describe-vpcs \
          --vpc-ids "${VPC_ID}" \
          --query "Vpcs[].CidrBlock" \
          --output text \
          --region "${REGION}")
echo CIDR_RANGE=${CIDR_RANGE}
```
Create an EFS security group and get its ID
```python
export SECURITY_GROUP_ID=$(aws ec2 create-security-group \
          --group-name "${CLUSTER_NAME}-EFS-SG" \
          --description "${CLUSTER_NAME} EFS security group" \
          --vpc-id "${VPC_ID}" \
          --output text)
echo SECURITY_GROUP_ID=${SECURITY_GROUP_ID}
```
Allow NFS access from the cluster VPC CIDR
```python
aws ec2 authorize-security-group-ingress \
          --group-id "${SECURITY_GROUP_ID}" \
          --protocol tcp \
          --port 2049 \
          --cidr "${CIDR_RANGE}"
```
Note the private subnet IDs of the VPC
```python
export PRIVATE_SUBNET_LIST=$(aws ec2 describe-subnets \
          --filters "Name=vpc-id,Values=${VPC_ID}" "Name=map-public-ip-on-launch,Values=false" \
          --query 'Subnets[*].SubnetId' \
          --output text)
echo PRIVATE_SUBNET_LIST=${PRIVATE_SUBNET_LIST}
```
For each subnet, create a mount target
```python
for SUBNET in ${PRIVATE_SUBNET_LIST}; do
    aws efs create-mount-target \
      --file-system-id "${FILE_SYSTEM_ID}" \
      --subnet-id "${SUBNET}" \
      --security-groups "${SECURITY_GROUP_ID}"
done
```

### Create the EFS Storage Class

{Download}`[Download the efs-sc.yaml config file template]<./efs-sc.yaml.template>`

Create the storage class config file from the downloaded template and the env variables and apply it
```python
envsubst < efs-sc.yaml.template > efs-sc.yaml
kubectl apply -f efs-sc.yaml
```
### Create an S3 bucket to store MLRun artifacts

```python
aws s3 mb s3://"${BUCKET_NAME}" --region "${REGION}"
```
### Create a Gateway Endpoint to access the bucket directly from the VPC 
Get the route-table IDs of the VPC
```python
TABLES_ARRAY=($(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=${VPC_ID}" --query 'RouteTables[*].RouteTableId' --output text))
echo TABLES_ARRAY=${TABLES_ARRAY[@]}
```
Create the Endpoint
```python
aws ec2 create-vpc-endpoint \
  --vpc-id ${VPC_ID} \
  --service-name com.amazonaws.${REGION}.s3 \
  --vpc-endpoint-type Gateway \
  --region ${REGION} \
  --route-table-ids ${TABLES_ARRAY[@]}
```
  
## Install the AWS Load Balancer Controller
```python
helm repo add eks https://aws.github.io/eks-charts
helm repo update
helm install aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system --set clusterName="${CLUSTER_NAME}" \
    --set serviceAccount.create=false \
    --set serviceAccount.name=aws-load-balancer-controller \
    --set vpcId=${VPC_ID}
```	
### Install MLRun CE with aws_values.yaml file into the mlrun namespace

{Download}`[Download the aws_values file template]<./aws_values.yaml.template>`

Create the aws_values.yaml config file from the downloaded template and the env variables
```python
envsubst < aws_values.yaml.template > aws_values.yaml
```

Add MLRun CE helm repo
```python
helm repo add mlrun-ce https://mlrun.github.io/ce
helm repo update
```
If you do not have a certificate, Run the following to install the MLRun CE helm chart using the values file
```python
helm upgrade --install --wait --dependency-update --version 0.6.3-rc28 --namespace mlrun -f aws_values.yaml mlrun-ce mlrun-ce/mlrun-ce
```
Alternatively, if you have a certificate, add the CERTIFICATE_ARN to the install command
```python
helm upgrade --install --wait --dependency-update --version 0.6.3-rc28 --namespace mlrun -f aws_values.yaml --set global.domainNameCertificate="<CERTIFICATE ARN>" mlrun-ce mlrun-ce/mlrun-ce
```
### Get the AWS Load Balancer CNAME and set is as a value for your DNS record
Configure the CNAME in your domain, pointing *\*.<cluster_name>.<domain>* to the Load Balancer URL:
```python
kubectl -n mlrun get ingress mlrun-ce-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```
### [Optional] Add access to the EKS cluster to additional users
To allow access to the EKS API for additional users, refer to the AWS documentation [Grant IAM users and roles access to Kubernetes APIs - Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/grant-k8s-access.html).

 

## Uninstalling the cluster and deleting the resources
Export the following env variables, Fill the desired <CLUSTER_NAME>, <DOMAIN_NAME>:
```python
export CLUSTER_NAME="<CLUSTER_NAME>"
export DOMAIN_NAME="<DOMAIN_NAME>"


export USER_NAME=$(aws iam get-user --query 'User.UserName' --output text)
export ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
export REGION=$(aws configure get region)
export CLUSTER_FQDN="${CLUSTER_NAME}.${DOMAIN_NAME}"
export BUCKET_NAME="${CLUSTER_NAME}-${ACCOUNT_ID}-bucket"
export ECR_REPO_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${CLUSTER_NAME}"
```

### Delete the S3 gateway endpoint
```python
ENDPOINT_IDS=$(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=${VPC_ID}" --query 'VpcEndpoints[*].VpcEndpointId' --output text)
for ENDPOINT_ID in ${ENDPOINT_IDS}; do
  echo "Deleting VPC Endpoint: ${ENDPOINT_ID}"
  aws ec2 delete-vpc-endpoints --vpc-endpoint-ids "${ENDPOINT_ID}"
done
```
### Delete the S3 bucket
```python
aws s3 rm s3://${BUCKET_NAME} --recursive
aws s3 rb s3://${BUCKET_NAME} --force
```
### Delete the EFS filesystem and mount targets

#### Get the filesystem ID

```python
FILE_SYSTEM_ID=$(aws efs describe-file-systems \
    --query "FileSystems[?Tags[?Key=='Cluster' && Value=='${CLUSTER_NAME}']].FileSystemId" \
    --output text)
echo FILE_SYSTEM_ID=${FILE_SYSTEM_ID}
```
#### Get the filesystem mount targets
```python
MOUNT_TARGET_IDS=$(aws efs describe-mount-targets \
  --file-system-id "${FILE_SYSTEM_ID}" \
  --query 'MountTargets[*].MountTargetId' \
  --output text)
echo MOUNT_TARGET_IDS=${MOUNT_TARGET_IDS}
```
#### Delete the mount targets
```python
for MOUNT_TARGET_ID in ${MOUNT_TARGET_IDS}; do
    aws efs delete-mount-target --mount-target-id "${MOUNT_TARGET_ID}"
done
```
### Get the EFS security group
```python
SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values="${CLUSTER_NAME}-EFS-SG" \
  --query "SecurityGroups[0].GroupId" \
  --output text)
echo SECURITY_GROUP_ID=${SECURITY_GROUP_ID}
```
### Delete the security group
```python
aws ec2 delete-security-group --group-id "${SECURITY_GROUP_ID}"
```
### Delete the file system
```python
aws efs delete-file-system --file-system-id "${FILE_SYSTEM_ID}"
```
### Delete the ECR repositories
```python

# Get all the repositories names
REPO_NAMES=$(aws ecr describe-repositories --region "${REGION}" --query 'repositories[?starts_with(repositoryName, `'${CLUSTER_NAME}'`)].repositoryName' --output text)
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
```
### Delete the EKS cluster
```python
eksctl delete cluster --name "${CLUSTER_NAME}"
```
### Delete the EBS volume leftovers

#### Get the relevant volume IDs
```python
VOLUME_IDS=$(aws ec2 describe-volumes --region "${REGION}" \
  --query "Volumes[?not_null(Tags[?Key=='Name']|[0].Value) && starts_with(Tags[?Key=='Name']|[0].Value, \`${CLUSTER_NAME}\`)].VolumeId" \
  --output text)
echo VOLUME_IDS=${VOLUME_IDS}
```
#### Delete the volumes
```python
for VOLUME_ID in ${VOLUME_IDS}; do
    aws ec2 delete-volume --volume-id ${VOLUME_ID} --region ${REGION}
done
```


