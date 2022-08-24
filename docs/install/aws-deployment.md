# Install MLRun on AWS
For AWS users, the easiest way to install MLRun is to use a native AWS deployment. This option deploys MLRun on an AWS EKS service using a cloud formation stack.

## Prerequisites
- AWS account <br>
  See [how to create a AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/).


> **Note that the MLRun software is free of charge, however, there is a cost for the AWS hardware and the EKS service.**


## Click the AWS icon to deploy Iguazio
<a href="https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/quickcreate?templateUrl=https%3A%2F%2Fmlrun-kit-alexp.s3.us-east-2.amazonaws.com%2Fquickstart-amazon-eks%2Ftemplates%2Figuazio-mlrun-kit-entrypoint-new-vpc.template.yaml&stackName=MLrun-community%20&param_AdditionalEKSAdminUserArn=&param_AvailabilityZones%5B%5D=&param_ClusterDomain=&param_DeployMLRunKit=true&param_EKSClusterName=&param_KeyPairName=&param_MLrunKitVersion=&param_NodeInstanceFamily=Standard&param_NodeInstanceType=m5.2xlarge&param_NumberOfAZs=3&param_NumberOfNodes=3&param_ProvisionBastionHost=Disabled&param_RegistryDomainName=index.docker.io&param_RegistryEmail=&param_RegistrySuffix=%2Fv1%2F&param_RegistryUsername=&param_RemoteAccessCIDR="><img src="../_static/images/aws-icon.png"></img></a>


After clicking the icon, the browser opens your AWS login page and directs you to the CloudFormation stack page. <br>
Fill in the form and click **Done**. MLRun creates a VPC with an EKS cluster and deploys all the services on top of it. <br>

The key components that are deployed on your EKS cluster are:
* EKS 
* MLRun server (including the feature store and MLRun graph)
* MLRun UI
* Kubeflow pipeline
* Real time serverless framework (Nuclio)
* Spark operator
* Jupyter lab

## Get started
Go to the output tab for the stack you created. You'll see the links for the MLRun UI, Jupyter, and the Kubeconfig command. <br>
It's helpful to go through the Quick start and the other tutorials as shown in the [documentation](../tutorial/index.html). Note that those tutorials and demos come 
built-in with Jupyter under the root folder of Jupyter.

## Configure the online feature store
The feature store can store data on a fast key-value database table for quick serving. This online feature store capability requires an external key-value database. 
Currently the MLRun feature store supports Iguazios' key-value database.
However, the MLRun roadmap includes support for the Redis database very soon as well as other key-value databases down the road. <br>
More information on the feature store can be found in [Feature store](../feature-store/feature-store.html).
    
## Integrate MLRun with Kafka for online serving   
Online serving use cases often require working with the MLRun graph and use a streaming engine for managing queues between steps and functions. 
MLRun supports Iguazio V3IO streams as well as Kafka streams. 
See the examples on how to configure the MLRun serving graph with [V3IO](serving/model-serving-get-started.html#v3io-stream-example) and [Kafka](../serving/model-serving-get-started.html#kafka-stream-example).
    

    
