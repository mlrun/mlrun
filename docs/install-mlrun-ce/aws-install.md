(aws-install)=
# Install MLRun CE on AWS

These instructions install the community edition (CE) on your Amazon EKS cluster. No pre-installation or pre-configuration is required. 

```{admonition} Note
These instructions install the community edition {{ ceversion }}, which currently includes the features in MLRun v{{ version }}.</br>
CE v0.10 will include the features in MLRun v1.10.
```

**In this section**
- [Prerequisites](#prerequisites)
- [IAM requirements](#iam-requirements)
- [Community Edition services](#community-edition-services)
- [Installation](#installation)
- [Uninstalling the cluster and deleting the resources](#uninstalling-the-cluster-and-deleting-the-resources)

## Prerequisites

- A registered domain name allowing wildcards with a dummy CNAME record (will be filled later with the AWS Load Balancer CNAME)
- AWS CLI is installed and configured. See [Installing or updating to the latest version of the AWS CLI - AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- eksctl is installed and configured. See [Installation - eksctl](https://eksctl.io/installation/) 
- kubectl is installed. See [Set up kubectl and eksctl - Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html)
- Helm is installed. See [Deploy applications with Helm on Amazon EKS - Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/helm.html)
- A bash shell to run the commands        


## IAM requirements
Verify that your AWS account has the following IAM policies.

````{dropdown} Expand to see the IAM policies:

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BasicServices",
            "Effect": "Allow",
            "Action": [
                "autoscaling:*",
                "cloudwatch:*",
                "elasticloadbalancing:*",
                "sns:*",
                "ec2:*",
                "s3:*",
                "s3-object-lambda:*",
                "eks:*",
                "elasticfilesystem:*",
                "cloudformation:*",
                "acm:*",
                "route53:*",
                "ecr:*"
            ],
            "Resource": "*"
        },
        {
            "Sid": "ServiceLinkedRoles",
            "Effect": "Allow",
            "Action": "iam:CreateServiceLinkedRole",
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "iam:AWSServiceName": [
                        "autoscaling.amazonaws.com",
                        "ec2scheduled.amazonaws.com",
                        "elasticloadbalancing.amazonaws.com",
                        "spot.amazonaws.com",
                        "spotfleet.amazonaws.com",
                        "transitgateway.amazonaws.com"
                    ]
                }
            }
        },
        {
            "Sid": "IAMPermissions",
            "Effect": "Allow",
            "Action": [
                "iam:AddRoleToInstanceProfile",
                "iam:TagRole",
                "iam:AttachRolePolicy",
                "iam:TagOpenIDConnectProvider",
                "iam:CreateInstanceProfile",
                "iam:CreateOpenIDConnectProvider",
                "iam:CreateRole",
                "iam:CreateServiceLinkedRole",
                "iam:DeleteInstanceProfile",
                "iam:DeleteOpenIDConnectProvider",
                "iam:DeleteRole",
                "iam:DeleteRolePolicy",
                "iam:DetachRolePolicy",
                "iam:GenerateServiceLastAccessedDetails",
                "iam:GetAccessKeyLastUsed",
                "iam:GetAccountPasswordPolicy",
                "iam:GetAccountSummary",
                "iam:GetGroup",
                "iam:GetInstanceProfile",
                "iam:GetLoginProfile",
                "iam:GetOpenIDConnectProvider",
                "iam:GetPolicy",
                "iam:GetPolicyVersion",
                "iam:GetRole",
                "iam:GetRolePolicy",
                "iam:GetServiceLastAccessedDetails",
                "iam:GetUser",
                "iam:ListAccessKeys",
                "iam:ListAccountAliases",
                "iam:ListAttachedGroupPolicies",
                "iam:ListAttachedRolePolicies",
                "iam:ListAttachedUserPolicies",
                "iam:ListGroupPolicies",
                "iam:ListGroups",
                "iam:ListGroupsForUser",
                "iam:ListInstanceProfilesForRole",
                "iam:ListMFADevices",
                "iam:ListOpenIDConnectProviders",
                "iam:ListPolicies",
                "iam:ListPoliciesGrantingServiceAccess",
                "iam:ListRolePolicies",
                "iam:ListRoles",
                "iam:ListRoleTags",
                "iam:ListSAMLProviders",
                "iam:ListSigningCertificates",
                "iam:ListUserPolicies",
                "iam:ListUsers",
                "iam:ListUserTags",
                "iam:PassRole",
                "iam:PutRolePolicy",
                "iam:RemoveRoleFromInstanceProfile",
                "kms:CreateGrant",
                "kms:CreateKey",
                "kms:Decrypt",
                "kms:DescribeKey",
                "kms:Encrypt",
                "kms:GenerateDataKeyWithoutPlaintext",
                "kms:GetKeyPolicy",
                "kms:GetKeyRotationStatus",
                "kms:ListResourceTags",
                "kms:PutKeyPolicy",
                "kms:ScheduleKeyDeletion",
                "kms:TagResource"
            ],
            "Resource": "*"
        },
        {
            "Sid": "AllowLanbda",
            "Effect": "Allow",
            "Action": [
                "lambda:CreateAlias",
                "lambda:CreateCodeSigningConfig",
                "lambda:CreateEventSourceMapping",
                "lambda:CreateFunction",
                "lambda:CreateFunctionUrlConfig",
                "lambda:Delete*",
                "lambda:Get*",
                "lambda:InvokeAsync",
                "lambda:InvokeFunction",
                "lambda:InvokeFunctionUrl",
                "lambda:List*",
                "lambda:PublishLayerVersion",
                "lambda:PublishVersion",
                "lambda:PutFunctionCodeSigningConfig",
                "lambda:PutFunctionConcurrency",
                "lambda:PutFunctionEventInvokeConfig",
                "lambda:PutProvisionedConcurrencyConfig",
                "lambda:TagResource",
                "lambda:UntagResource",
                "lambda:UpdateAlias",
                "lambda:UpdateCodeSigningConfig",
                "lambda:UpdateEventSourceMapping",
                "lambda:UpdateFunctionCode",
                "lambda:UpdateFunctionCodeSigningConfig",
                "lambda:UpdateFunctionConfiguration",
                "lambda:UpdateFunctionEventInvokeConfig",
                "lambda:UpdateFunctionUrlConfig"
            ],
            "Resource": "*"
        },
        {
            "Sid": "CertificateService",
            "Effect": "Allow",
            "Action": "iam:CreateServiceLinkedRole",
            "Resource": "arn:aws:iam::*:role/aws-service-role/acm.amazonaws.com/AWSServiceRoleForCertificateManager*",
            "Condition": {
                "StringEquals": {
                    "iam:AWSServiceName": "acm.amazonaws.com"
                }
            }
        },
        {
            "Sid": "DeleteRole",
            "Effect": "Allow",
            "Action": [
                "iam:DeleteServiceLinkedRole",
                "iam:GetServiceLinkedRoleDeletionStatus",
                "iam:GetRole"
            ],
            "Resource": "arn:aws:iam::*:role/aws-service-role/acm.amazonaws.com/AWSServiceRoleForCertificateManager*"
        },
        {
            "Sid": "SSM",
            "Effect": "Allow",
            "Action": [
                "logs:*",
                "ssm:AddTagsToResource",
                "ssm:GetParameter",
                "ssm:DeleteParameter",
                "ssm:PutParameter",
                "cloudtrail:GetTrail",
                "cloudtrail:ListTrails"
            ],
            "Resource": "*"
        }
    ]
}
```
````

## Community Edition services

The MLRun CE (Community Edition) includes the following services:
- MLRun - https://github.com/mlrun/mlrun
  - MLRun API
  - MLRun UI
  - MLRun DB (MySQL)
- Nuclio - https://github.com/nuclio/nuclio
- Jupyter - https://github.com/jupyter/notebook (+MLRun integrated)
- Kafka
- TimescaleDB
- Prometheus stack - https://github.com/prometheus-community/helm-charts
  - Prometheus
  - Grafana  

## Installation
1. [Optional] Create or import a certificate to AWS Certificate Manager for the relevant domain including wildcards **\*.SYSTEM_NAME.example.com** by one of:
    - Create a certificate: [AWS Certificate Manager public certificates - AWS Certificate Manager](https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-request-public.html)
	- Import an existing certificate: [Import a certificate - AWS Certificate Manager](https://docs.aws.amazon.com/acm/latest/userguide/import-certificate-api-cli.html)</br>
   Note the ARN of the certificate.
2. Export the following env variables, fill in the relevant <SYSTEM_NAME> and <DOMAIN_NAME>:
   ```
   export SYSTEM_NAME="<SYSTEM_NAME>"
   export DOMAIN_NAME="<DOMAIN_NAME>"
   ```
2. Export a comma-delimited list of CIDR ranges that will be able to access the MLRun services via the AWS ALB:
   ```
   export INBOUND_CIDRS="<CIDR_RANGE>[,<CIDR_RANGE>].."
   ```
   Ensure the CIDR_RANGE is correctly formatted, including the subnet mask (e.g. 192.168.1.0/24).
2. Export the remaining derived values:
   ```
   export USER_NAME=$(aws iam get-user --query 'User.UserName' --output text)
   export ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
   export REGION=$(aws configure get region)
   export SYSTEM_FQDN="${SYSTEM_NAME}.${DOMAIN_NAME}"
   export BUCKET_NAME="${SYSTEM_NAME}-${ACCOUNT_ID}-bucket"
   export ECR_REPO_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${SYSTEM_NAME}"
   ```
2. Create the EKS cluster:
    1. Download the {download}`EKS config file <./cluster.yaml.template>`.
    2. Create an EKS `cluster.yaml` config file from the downloaded template and the env variables:
	   ```
   	   envsubst < cluster.yaml.template > cluster.yaml
	   ```
	   The minimal instance size required for MLRun to operate is m5.xlarge. You can increase the instance numbers and sizes in the `cluster.yaml` per your requirements.
	2. Create an EKS cluster using the cluster.yaml conifg file:
	   ```
	   eksctl create cluster -f cluster.yaml
	   ```
	The installation also creates the mlrun namespace and add IAM roles, policies and service accounts for EBS and S3 access.
2. Configure EBS as the default Storage Class:
   ```
   kubectl patch storageclass gp2 -p '{"metadata": {"annotations": {"storageclass.kubernetes.io/is-default-class": "true"}}}'
   ```
2. Create an S3 bucket to store MLRun artifacts:
   ```
   aws s3 mb s3://"${BUCKET_NAME}" --region "${REGION}"
   ```
2. Get the cluster’s VPC ID:
      ```
   export VPC_ID=$(aws eks describe-cluster \
             --name "${SYSTEM_NAME}" \
             --query "cluster.resourcesVpcConfig.vpcId" \
             --output text)
   echo VPC_ID=${VPC_ID}
   ```
2. Create a Gateway Endpoint to access the bucket directly from the VPC:
   1. Get the route-table IDs of the VPC:
      ```
      TABLES_ARRAY=($(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=${VPC_ID}" --query 'RouteTables[*].RouteTableId' --output text))
      echo TABLES_ARRAY=${TABLES_ARRAY[@]}
      ```
   2. Create the endpoint:
      ```
      aws ec2 create-vpc-endpoint \   
      --vpc-id ${VPC_ID} \   
      --service-name com.amazonaws.${REGION}.s3 \   
      --vpc-endpoint-type Gateway \
      --region ${REGION} \
      --route-table-ids ${TABLES_ARRAY[@]}
      ```
2. Install the AWS Load Balancer Controller:
   ```
   helm repo add eks https://aws.github.io/eks-charts
   helm repo update
   helm install aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system --set clusterName="${SYSTEM_NAME}" \
   --set serviceAccount.create=false \
   --set serviceAccount.name=aws-load-balancer-controller-sa \
   --set vpcId=${VPC_ID}
   ```
2. Install the MLRun CE with `aws_values.yaml` file into the mlrun namespace:
   1. Download the {download}`aws_values file template <./aws_values.yaml.template>`.
   2. Create the `aws_values.yaml` config file from the downloaded template and the env variables:
      ```
      envsubst < aws_values.yaml.template > aws_values.yaml
      ```
   2. Add the MLRun CE helm repo:
      ```
      helm repo add mlrun-ce https://mlrun.github.io/ce
      helm repo update
      ```
   3. If you do not have a certificate, install the MLRun CE helm chart by running this, using the values file:
      ``` 
      helm install --wait --dependency-update --namespace mlrun -f aws_values.yaml mlrun-ce mlrun-ce/mlrun-ce --version 0.7.0
      ```
   4. Alternatively, if you have a certificate, add the CERTIFICATE_ARN to the install command:
      ```
      helm install --wait --dependency-update --namespace mlrun -f aws_values.yaml --set global.domainNameCertificate="<CERTIFICATE ARN>" mlrun-ce mlrun-ce/mlrun-ce --version 0.7.0
      ```
2. Get the AWS Load Balancer CNAME and set it as a value for your DNS record. Configure the CNAME in your domain, pointing **\*.<system_name>.\<domain>** to the Load Balancer URL:
   ```
   kubectl -n mlrun get ingress mlrun-ce-ingress -o custom-columns=":status.loadBalancer.ingress[0].hostname" --no-headers
   ```
2. To work with Spark and PIA authentication
   ```
   my_func.spec.spark_conf['spark.hadoop.fs.s3a.aws.credentials.provider'] = 'com.amazonaws.auth.WebIdentityTokenCredentialsProvider'
   ```
2. To configure Spark to work with SeaweedFS:
   ```
   func.spec.spark_conf["spark.hadoop.fs.s3a.endpoint"] = 'http://seaweedfs-s3.<"namespace-name">.svc.cluster.local:8333' 
   func.spec.spark_conf['spark.hadoop.fs.s3a.path.style.access'] = 'true'
   ```

2. [Optional] Add access to the EKS API for additional users. See: [Grant IAM users and roles access to Kubernetes APIs - Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/grant-k8s-access.html).
2. [Optional] Grant access to the S3 bucket for additional users. See the AWS walkthrough example: [Bucket owner granting its users bucket permissions - Amazon Simple Storage Service](https://docs.aws.amazon.com/AmazonS3/latest/userguide/example-walkthroughs-managing-access-example1.html).

## Uninstalling the cluster and deleting the resources
1. Export the following env variables; fill in the relevant <SYSTEM_NAME>, <DOMAIN_NAME>:
   ```
   export SYSTEM_NAME="<SYSTEM_NAME>"
   export DOMAIN_NAME="<DOMAIN_NAME>"
   
   export USER_NAME=$(aws iam get-user --query 'User.UserName' --output text)
   export ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
   export REGION=$(aws configure get region)
   export SYSTEM_FQDN="${SYSTEM_NAME}.${DOMAIN_NAME}"
   export BUCKET_NAME="${SYSTEM_NAME}-${ACCOUNT_ID}-bucket"
   export ECR_REPO_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${SYSTEM_NAME}"
   ```
2. Get the cluster’s VPC ID:
   ```
   export VPC_ID=$(aws eks describe-cluster \
             --name "${SYSTEM_NAME}" \
             --query "cluster.resourcesVpcConfig.vpcId" \
             --output text)
   echo VPC_ID=${VPC_ID}
   ```
2. Delete the S3 gateway endpoint:
   ```
   NDPOINT_IDS=$(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=${VPC_ID}" --query 'VpcEndpoints[*].VpcEndpointId' --output text)
   for ENDPOINT_ID in ${ENDPOINT_IDS}; do
     echo "Deleting VPC Endpoint: ${ENDPOINT_ID}"
     aws ec2 delete-vpc-endpoints --vpc-endpoint-ids "${ENDPOINT_ID}"
   done
   ```
2. Delete the S3 bucket:
   ```
   aws s3 rm s3://${BUCKET_NAME} --recursive
   aws s3 rb s3://${BUCKET_NAME} --force
    ```
2. Delete the ECR repositories:
   ```
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
   ```
2. Delete the EKS cluster:<br>
   ```
   eksctl delete cluster --name "${SYSTEM_NAME}"
   ```
2. Delete the EBS volume leftovers.
   1. Get the relevant volume IDs:
      ```
      VOLUME_IDS=$(aws ec2 describe-volumes --region "${REGION}" \
      --query "Volumes[?not_null(Tags[?Key=='Name']|[0].Value) && starts_with(Tags[?Key=='Name']|[0].Value, \`${SYSTEM_NAME}\`)].VolumeId" \
      --output text)
      echo VOLUME_IDS=${VOLUME_IDS}
      ```
   2. Delete the volumes:
      ```
      for VOLUME_ID in ${VOLUME_IDS}; do
          aws ec2 delete-volume --volume-id ${VOLUME_ID} --region ${REGION}
      done
      ```

