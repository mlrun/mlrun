(model-training)=
# Model Training

There are many methods and techniques for training a machine learning model. However, there is much more to model training than the training code itself.

MLRun provides MLOps functionality for model training including job orchestration, experiment tracking, creating reusable components, and distributed training out of the box. Please see below for information on each of these capabilies:

## Jobs

Training a model is an example of a `Job` — something that is run once to completion. MLRun provides you with a convenient syntax to take a Python training script and automatically package and deploy it on top of a production-ready Kubernetes cluster. You can also configure many aspects of this deployment including [CPU/MEM/GPU resources](https://github.com/mlrun/mlrun/pull/2166/files/runtimes/configuring-job-resources.html), Python dependencies, [distributed runtimes](https://github.com/mlrun/mlrun/pull/2166/files/..runtimes/distributed.html), and more.

See the [Create a Basic Training Job](https://github.com/mlrun/mlrun/pull/2166/training/create-a-basic-training-job.html) page for an example of a simple training job. Additionally, see [Managing Job Resources](https://github.com/mlrun/mlrun/pull/2166/runtimes/configuring-job-resources.html) for ways to configure your jobs.

## Logging Artifacts

While training your model, there may be things you want to log including the model itself, datasets, plots/charts, metrics, etc. All of this and more can be tracked using MLRun experiment tracking.

MLRun supports automatic logging for major ML frameworks such as sklearn, PyTorch, TensorFlow, LightGBM, etc. MLRun also supports manually logging models, datasets, metrics, and more.

See the [Working with Data and Model Artifacts](https://github.com/mlrun/mlrun/pull/2166/training/working-with-data-and-model-artifacts.html) page for an example.

## Function Marketplace

In addition to running your own Python code, you can also utilize work that others have done by importing from the [MLRun Function Marketplace](https://www.mlrun.org/marketplace/). There are many reusable functions for data preparation, data analysis, model training, model deployment, and more.

For example, you can leverage the power of AutoML by using the [Auto Trainer](https://www.mlrun.org/marketplace/functions/master/auto_trainer/latest/example/) function or perform automated Exploratory Data Analysis (EDA) by using the [Describe](https://www.mlrun.org/marketplace/functions/master/describe/latest/example/) function.

See the [MLRun Functions Marketplace](https://github.com/mlrun/mlrun/pull/2166/runtimes/load-from-marketplace.html) page for an example.

## Distributed Training

MLRun also allows you to utilize distributed training and computation frameworks out of the box such as Spark, Dask, and Horovod. These are useful when your data does not fit into memory, you want to do computations in parallel, or you want to leverage multiple physical machines to train your model.

See the [Spark](https://github.com/mlrun/mlrun/pull/2166/runtimes/spark-operator.html), [Dask](https://github.com/mlrun/mlrun/pull/2166/runtimes/dask-overview.html), and [Horovod](https://github.com/mlrun/mlrun/pull/2166/runtimes/horovod.html) pages respectively for examples.