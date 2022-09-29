(deploy-models-and-ml-applications)=

# Deploy models and ML applications

One of the advantages of using MLRun, is simplifying the deployment process. Deployment is more than just model
deployment. Models usually run as part of a greater system which requires data processing before and after
executing the model as well as being part of a business application.

Generally, there are two main modes of deployment:

1. **Model serving**: this is the process of having models respond for real-time events. The challenge here is
  usually ensuring that the data processing is performed in the same way that the batch training was done and
  sending the response in low latency. MLRun includes a specialized serving graph that eases that creation
  of a data transformation pipeline as part of the model serving. Feature store support is another way of
  ensuring feature calculation remain consistent between the training process and the serving process. For
  an end-to-end demo of model serving, refer to the
  [**Serving pre-trained ML/DL models tutorial**](../tutorial/03-model-serving.html).
2. **Batch inference**: this includes a process that runs on a large dataset. The data is usually read from
  an offline source, such as files or databases and the result is also written to offline targets. It is common
  to set-up a schedule when running batch inference. For an end-to-end demo of batch inference, refer to the
  [**batch inference and drift detection tutorial**](../tutorial/07-batch-infer.html).

**In this section**

```{toctree}
:maxdepth: 1

../serving/serving-overview
../feature-store/training-serving
../deployment/batch_inference
../serving/canary
```
