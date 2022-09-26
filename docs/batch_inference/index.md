(batch_inference_overview)=
# Batch inference

Batch inference or offline inference addresses the need to run machine learning model on large datasets.

It is the process of generating outputs on a batch of observations. The batch runs are typically generated during some recurring schedule (e.g., hourly, or daily). These inferences are then stored in a database or a file and can be made available to developers or end users. Batch inference may sometimes take advantage of big data technologies such as Spark to generate predictions. This allows data scientists and machine learning engineers to take advantage of scalable compute resources to generate many predictions at once. With batch inference, the goal is usually tied to time constraints and the service-level agreement (SLA) of the job. Conversely, in real time serving, the goal is usually to optimize the number of transactions per second that the model can process. In addition, the output of batch inference goes to a file or a table in a database while an online application displays a result to the user.

Check out how to {ref}`test and deploy batch model inference<test_and_deploy_batch_model_inference>` and the {ref}`built-in batch inference function<using_built-in_batch_inference>`. You should also check out the [batch inference tutorial](../tutorial/07-batch-infer.ipynb). 

```{toctree}
:maxdepth: 1
:hidden: true

test_and_deploy_batch_model_inference
using_built-in_batch_inference
```