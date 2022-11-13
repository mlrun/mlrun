(install-on-aws)=
# Install MLRun on AWS

For AWS users, the easiest way to install MLRun is to use a native AWS deployment. This option deploys MLRun on an AWS EKS service using a cloud formation stack.

## Prerequisites

AWS account with Permission to run a CloudFormation stack, create an EKS cluster, Create EC2 instances, define VPC, create S3 buckets and deploy and pull images from ECR.

  For more information, see [how to create a new AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/).

- To access instance management, you will need to have at least one key pair for SSH keys. For more information see [Amazon EC2 key pairs and Linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html).
- You need to have a route53 domain configured. External domain registration is currently not supported. For more information see [What is Amazon Route 53](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/Welcome.html).


> **Note**:
> - The MLRun software is free of charge, however, there is a cost for the AWS infrastructure services such as EKS, EC2, S3 and ECR. The actual pricing depends on a large set of factors including, for example, the region, the number of EC2 instances, the amount of storage consumed, and the data transfer costs. Other factors include, for example,  reserved instance configuration, saving plan, and AWS credits you have associated with your account.
> - It is recommended to use the [AWS pricing calculator](https://calculator.aws) to calculate the expected cost, as well as the [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/) to manage the cost, monitor and set-up alerts.

## Post deployment expectations

The key components deployed on your EKS cluster are:

- MLRun server (including the feature store and the MLRun graph)
- MLRun UI
- Kubeflow pipeline
- Real time serverless framework (Nuclio)
- Spark operator
- Jupyter lab
- Grafana

## Configuration settings

Make sure you are logged in to the correct AWS account.

**Click the AWS icon to deploy MLRun.**

<a href="https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/quickcreate?templateUrl=https%3A%2F%2Fmlrun-ce-cfn.s3.us-east-2.amazonaws.com%2Fquickstart-amazon-eks%2Ftemplates%2Figuazio-mlrun-kit-entrypoint-new-vpc.template.yaml&stackName=MLrun-community&param_AdditionalEKSAdminUserArn=&param_AvailabilityZones%5B%5D=&param_ClusterDomain=&param_DeployMLRunKit=true&param_EKSClusterName=&param_KeyPairName=&param_MLrunKitVersion=&param_NodeInstanceFamily=Standard&param_NodeInstanceType=m5.2xlarge&param_NumberOfAZs=3&param_NumberOfNodes=3&param_ProvisionBastionHost=Disabled&param_RegistryDomainName=index.docker.io&param_RegistryEmail=&param_RegistrySuffix=%2Fv1%2F&param_RegistryUsername=&param_RemoteAccessCIDR="><img src="../_static/images/aws_launch_stack.png"></img></a>

After clicking the icon, the browser directs you to the CloudFormation stack page in your AWS account, or redirects you to the AWS login page if you are not currently logged in.

> **Note**: You must fill in fields marked as mandatory (m) for the configuration to complete. Fields marked as optional (o) can be left blank.

1. **Stack name** (m)&mdash;the name of the stack. You cannot continue if left blank. This field becomes the logical id of the stack. Stack name can include letters (A-Z and a-z), numbers (0-9), and dashes (-).

**Parameters**

2. **EKS cluster name** (m)&mdash;the name of EKS cluster created. The EKS cluster is used to run the MLRun services. For example, `https://jupyter.<eks_cluster_name>.<route53_domain_name>`

**VPC network Configuration**

3. **Number of Availability Zones** (m)&mdash;number of availability zones. The default is set to 3. Choose from the dropdown to change the number. The minimum is 2.
4. **Availability zones** (m)&mdash;select a zone from the dropdown. The list is based on the region of the instance. The number of zones must match the number of zones Number of Availability Zones.
5. **Allowed external access CIDR** (m)&mdash;range of IP address allowed to access the cluster. Addresses that are not in this range are will not be able to access the cluster.

**Amazon EKS configuration**

6. **Additional EKS admin ARN (IAM user)** (o)&mdash;add an additional admin user to the instance. Users can be added after the stack has been created. For more information see [Create a kubeconfig for Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/create-kubeconfig.html).
7. **Instance type** (m)&mdash;select from the dropdown list. The default is m5.2xlarge. For size considerations see [Amazon EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/).
8. **Number of Nodes** (m)&mdash;number of nodes in the cluster. The minimum must match the number of **Availability Zones**. The number of nodes combined with the **Instance type** determines the AWS infrastructure cost. 

**Amazon EC2 configuration**

9. **SSH key name** (m)&mdash;select from the stored keys in the dropdown. The list is based on the SSH keys that are in your account. For more information about SSH Keys see [Amazon EC2 key pairs and Linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html).
10. **Provision bastion host** (m)&mdash;create a bastion host for SSH access to the Kubernetes nodes. The default is disabled.


**Iguazio MLRun configuration**

11. **Route 53 hosted DNS domain** (m)&mdash;enter the name of your registered Route53 domain. **Note:** Only route53 domains are acceptable.
12. **The URL of your REDIS database** (o)&mdash;the URL of your Redis database. This is only required if you're using Redis with the online feature store. See [how to configure the online feature store](#configure-online-feature-store) for more details.

**Other parameters**

13. **MLrunCeVersion** (m)&mdash;the MLRun Community Edition version to install. Leave the default value for the latest CE release.

**Capabilities**

14. Check all the capabilities boxes (m).

Press **Create Stack** to continue the deployment.
The stack creates a VPC with an EKS cluster and deploys all the services on top of it.

> **note**: It could take up to 2 hours for your stack to be created.

## Getting started
When the stack is complete, go to the **output** tab for the stack you created. There are links for the MLRun UI, Jupyter and the Kubeconfig command.

It's recommended to go through the quick-start and the other tutorials as shown in the documentation. These tutorials and demos come built-in with Jupyter under the root folder of Jupyter.

## AWS services to be aware of

When installing the MLRun Community Edition via Cloud Formation, several storage resources are created, some of which persist even after uninstalling the stack:

- **PVs via AWS storage provider**: Used to hold the file system of the stacks pods, including the MySQL database of MLRun. These are deleted when the stack is uninstalled.
- **S3 Bucket**: A bucket named `<EKS cluster name>-<Random string>` is created in the AWS account that installs the stack (where `<EKS cluster name>` is the name of the EKS cluster you chose and `<Random string>` is part of the CloudFormation stack ID). The bucket is used for MLRun’s artifact storage, and is not deleted when uninstalling the stack. The user must empty the bucket and delete it.
- **Container Images in ECR**: When building and deploying MLRun and Nuclio functions via the MLRun Community Edition, the function images are stored in an ECR belonging to the AWS account that installs the stack. These images persist in the account’s ECR and are not deleted either.

<a id="configure-online-feature-store"/>

## How to configure the online feature store

The feature store can store data on a fast key value database table for quick serving. This online feature store capability requires an external key value database.

Currently the MLRun feature store supports the following options:
- Redis 
- Iguazio key value database
  
To use Redis, you must install Redis separately and provide the Redis URL when configuring the AWS CloudFormation stack. Refer to the [Redis getting-started page](https://redis.io/docs/getting-started/) for information about Redis installation.

## Streaming support

For online serving, it is often convenient to use MLRun graph with a streaming engine. This allows managing queues between steps and functions. 
MLRun supports Kafka streams as well as Iguazio V3IO streams. 
See the examples on how to configure the MLRun serving graph with {ref}`kafka<serving-kafka-stream-example>` and {ref}`V3IO<serving-v3io-stream-example>`.

## Cleanup

To free up the resources used by MLRun:

- Delete the stack. See [instructions for deleting a stack on the AWS CloudFormation console](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-delete-stack.html) for more details.
- Delete the S3 that begins with the same name as your EKS cluster`.
- Delete any remaining images in ECR.

You may also need to check any external storage that you used.
