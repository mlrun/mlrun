(model-training)=
# Model Training

There are many methods and techniques for training a machine learning model. However, there is much more to model training than the training code itself.

MLRun provides MLOps functionality for model training including job orchestration, experiment tracking, creating reusable components, and distributed training out of the box. Please see below for information on each of these capabilies:

### Jobs

Training a model is an example of a `Job` - something that is run once to completion. MLRun provides you with a convenient syntax to take a Python training script and automatically package and deploy on top of a production ready Kubernetes cluster. You can also configure many aspects of this deployment including CPU/MEM/GPU resources, Python dependencies, distributed runtimes, and more.

See the [Create a Basic Training Job](#) page for an example of a simple training job. Additionally, see [Managing Job Resources](https://docs.mlrun.org/en/latest/runtimes/configuring-job-resources.html) for ways to configure your jobs.

### Logging Artifacts

While training your model, there may be things you want to log including the model itself, datasets, plots/charts, metrics, etc. All of this and more can be tracked using MLRun experiment tracking.

There is support for automatic logging for major ML frameworks such as sklearn, PyTorch, TensorFlow, LightGBM, etc. There is also support for manually logging models, datasets, metrics, and more.

See the the [Working with Data and Model Artifacts](#) page for an example.

### Function Marketplace

In addition to running your own Python code, you can also utilize work others have done by importing from the [MLRun Function Marketplace](https://www.mlrun.org/marketplace/). There are many reusable functions for data preperation, data analysis, model training, model deployment, and more.

For example, you can leverage the power of AutoML by using the [Auto Trainer](https://www.mlrun.org/marketplace/functions/master/auto_trainer/latest/example/) function or perform automated Exploratory Data Analysis (EDA) by using the [Describe](https://www.mlrun.org/marketplace/functions/master/describe/latest/example/) function.

See the [MLRun Functions Marketplace](https://docs.mlrun.org/en/latest/runtimes/load-from-marketplace.html) page for an example.

### Distributed Training

MLRun also allows you to utilize distributed training and computation frameworks out of the box such as Spark, Dask, and Horovod. These are useful when your data does not fit into memory, you want to do computations in parallel, or you want to leverage multiple physical machines for to train your model.

See the [Spark](https://docs.mlrun.org/en/latest/runtimes/spark-operator.html), [Dask](https://docs.mlrun.org/en/latest/runtimes/dask-overview.html), and [Horovod](https://docs.mlrun.org/en/latest/runtimes/horovod.html) pages respectively for examples.