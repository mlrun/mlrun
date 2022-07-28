# Install MLRun on AWS
For an AWS user the easiest way to install MLRun is to use a native AWS deployment. This option deploys MLrun on AWS EKS using a cloud formation.

## Prerequisites
- AWS account 
- EKS cluster 1.22 

## Click on the AWS icon to deploy Iguazio
<a href="https://console.aws.amazon.com/"><img src="_static/images/aws-icon.png"></img></a>

Once clicked, the browser opens your AWS login page and direct you to the cloud formation stack page <br>
Next, you need to fill in the form and once click on Done MLRun would be deployed on top of your EKS cluster <br>

The key components that being deployed on your EKS cluster are:

* MLRun server (including feature store and MLrun graph)
* MLRun UI
* Kubeflow pipeline
* Real time serverless framework  (Nuclio)
* Jupyter lab

## Feature store
### How to configure the online feature store

The feature store has an online part for quick serving. The online feature requires a key value database for fast retrieval . <br>
Given that, in order to use the online featre set you need to work with Redis database (Additional databases will be supported soon) <br>
For redis - you can work with their managed redis service such as https://redis.com/ or install the open source version https://github.com/redis/redis <br>
To learn how to configure the feature store with Redis go to <TBD LINK>
    
## Online processing 
### How to integrate MLRun with kafka for online processing
    
    
Online serving use cases often require working with MLRun graph and uses a streaming engine for storing queues between steps and functions. <br>
MLRun supports Iguazio V3IO stream as well as kafka stream. <br>
Here you can find examples on how to configure MLrun working with kafka <TBD Link>
