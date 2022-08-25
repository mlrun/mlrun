# Install MLRun on AWS
For an AWS user the easiest way to install MLRun is to use a native AWS deployment. This option deploys MLrun on AWS EKS service using a cloud formation stack.

## Prerequisites
- AWS account <br>
  [how to create a new AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)


> **Note that the MLRun software is free of charge ,however, there is a cost for the AWS hardware and the EKS service.**


## Click on the AWS icon to deploy Iguazio
<a href="https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/quickcreate?templateUrl=https%3A%2F%2Fmlrun-kit-alexp.s3.us-east-2.amazonaws.com%2Fquickstart-amazon-eks%2Ftemplates%2Figuazio-mlrun-kit-entrypoint-new-vpc.template.yaml&stackName=MLrun-community%20&param_AdditionalEKSAdminUserArn=&param_AvailabilityZones%5B%5D=&param_ClusterDomain=&param_DeployMLRunKit=true&param_EKSClusterName=&param_KeyPairName=&param_MLrunKitVersion=&param_NodeInstanceFamily=Standard&param_NodeInstanceType=m5.2xlarge&param_NumberOfAZs=3&param_NumberOfNodes=3&param_ProvisionBastionHost=Disabled&param_RegistryDomainName=index.docker.io&param_RegistryEmail=&param_RegistrySuffix=%2Fv1%2F&param_RegistryUsername=&param_RemoteAccessCIDR="><img src="../_static/images/aws-icon.png"></img></a>


Once clicked, the browser opens your AWS login page and direct you to the cloud formation stack page <br>
Next, you need to fill in the form and once click on Done we'll craete a new VPC with an EKS cluster and deploy all the services on top of it. <br>

The key components that being deployed on your EKS cluster are:

* EKS 
* MLRun server (including feature store and MLrun graph)
* MLRun UI
* Kubeflow pipeline
* Real time serverless framework  (Nuclio)
* Spark operator
* Jupyter lab

## How to get started?
Once the deployment is done go to the output tab for the satck you've created. You'll see the links for the MLRun UI, Jupyter and the Kubeconfig command <br>
We encourgae you to go through the quick start and the other tutorials as shown in the documentaion. note that those tutorials and demos comes built-in with Jupyter under the root folder of Jupyter.


## AWS services to be aware of
Note that when installing the MLRun Community Edition via Cloud Formation, several storage resources are created, some of which will persist even after uninstalling the stack:
* PVs via AWS storage provider: Used to hold the file system of the stacks pods, including the MySQL database of MLRun. These will be deleted once the stack is uninstalled.
* S3 Bucket: A bucket named mlrun will be created in the AWS account which installs the stack. The bucket is used for MLRun’s artifact storage, and will not be deleted when uninstalling the stack. The user will be required to empty the bucket and delete it.
* Container Images in ECR: When building and deploying MLRun and Nuclio functions via the MLRun Community Edition, the function images will be stored in an ECR belonging to the AWS account which installs the stack. These images will persist in the account’s ECR and will not be deleted either.

## Feature store
### How to configure the online feature store

The feature store can store data on a fast key value database table for quick serving. This online feature store capability requires an external key value database. <br>
Currently the MLrun feature store supports Iguazios' key value database.<br>
However, in our roadmap we plan to support redis database very soon as well as other key value databases down the road. <br>
More information on the feature store can be found here [Feature store doc](https://docs.mlrun.org/en/latest/feature-store/feature-store.html)
    
## Online serving 
### How to integrate MLRun with kafka for online serving
    
    
Online serving use cases often require working with MLRun graph and uses a streaming engine for managing queues between steps and functions. <br>
MLRun supports Iguazio V3IO stream as well as kafka stream. <br>
Here you can find examples on how to configure MLrun serving graph with kafka <TBD Link>
[Model serving with kafka](https://github.com/mlrun/mlrun/blob/5265121b44b35b0ccc8dbf0430a22d19860cb1c3/docs/serving/model-serving-get-started.ipynb)
    

