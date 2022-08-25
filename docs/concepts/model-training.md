(model-training)=
# Model training

There are many methods and techniques for training a machine learning model. However, there is much more to model training than the training code itself.

MLRun provides MLOps functionality for model training including job orchestration, experiment tracking, creating reusable components, and distributed training out of the box. Continue reading for information on each of these capabilies:

## Jobs

Training a model is an example of a `Job` — something that is run once to completion. MLRun provides you with a convenient syntax to take a Python training script and automatically package and deploy it on top of a production-ready Kubernetes cluster. You can also configure many aspects of this deployment including {ref}`configuring-job-resources`, Python dependencies, {ref}`distributed-functions`, and more.

See the {ref}`create-a-basic-training-job` page for an example of a simple training job. Additionally, see {ref}`configuring-job-resources` for ways to configure your jobs.

## Logging artifacts

While training your model, there may be things you want to log including the model itself, datasets, plots/charts, metrics, etc. All of this and more can be tracked using MLRun experiment tracking.

MLRun supports automatic logging for major ML frameworks such as sklearn, PyTorch, TensorFlow, LightGBM, etc. MLRun also supports manually logging models, datasets, metrics, and more.

See {ref}`working-with-data-and-model-artifacts` for an example.

## Function Marketplace

In addition to running your own Python code, you can also utilize work that others have done by importing from the [MLRun Function Marketplace](https://www.mlrun.org/marketplace/). There are many reusable functions for data preparation, data analysis, model training, model deployment, and more.

For example, you can leverage the power of AutoML by using the {ref}`built-in training function<using-built-in-training-function>` or perform automated Exploratory Data Analysis (EDA) by using the [Describe](https://www.mlrun.org/marketplace/functions/master/describe/latest/example/) function.

## Distributed Training

MLRun also allows you to utilize distributed training and computation frameworks out of the box such as Spark, Dask, and Horovod. These are useful when your data does not fit into memory, you want to do computations in parallel, or you want to leverage multiple physical machines to train your model.

See the {ref}`Spark <spark-operator>`, {ref}`Dask <dask-overview>`, and {ref}`Horovod <horovod>` pages respectively for examples.