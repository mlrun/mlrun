(auto-logging-mlops)=
# Automated experiment tracking

You can write custom training functions or use built-in hub functions for training models using 
common open-source frameworks and/or cloud services (such as AzureML, Sagemaker, etc.). 

Inside the ML function you can use the `apply_mlrun()` method, which automates the tracking and MLOps
functionality.

With `apply_mlrun()` the following outputs are generated automatically:
* Plots &mdash; loss convergence, ROC, confusion matrix, feature importance, etc.
* Metrics &mdash; accuracy, loss, etc.
* Dataset artifacts &mdash; like the dataset used for training and / or testing
* Custom code &mdash; like custom layers, metrics, and so on
* Model artifacts &mdash; enables versioning, monitoring and automated deployment

In addition it handles automation of various MLOps tasks like scaling runs over multiple containers 
(with Dask, Horovod, and Spark), run profiling, hyperparameter tuning, ML Pipeline, and CI/CD integration, etc.

`apply_mlrun()` accepts the model object and various optional parameters. For example:

```python
apply_mlrun(model=model, model_name="my_model", 
            x_test=x_test, y_test=y_test)
```

When specifying the `x_test` and `y_test` data it generates various plots and calculations to evaluate the model.
Metadata and parameters are automatically recorded (from the MLRun `context` object) and don't need to be specified.

`apply_mlrun` is framework specific and can be imported from [MLRun's **frameworks**](../api/mlrun.frameworks/index.html) 
package &mdash; a collection of commonly used machine and deep learning frameworks fully supported by MLRun.

`apply_mlrun` can be used with its default settings, but it is highly flexible and rich with different options and 
configurations. Reading the docs of your favorite framework to get the most out of MLRun:
- [SciKit-Learn](../api/mlrun.frameworks/mlrun.frameworks.sklearn.html)
- [TensorFlow (and Keras)](../api/mlrun.frameworks/mlrun.frameworks.tf_keras.html)
- [PyTorch](../api/mlrun.frameworks/mlrun.frameworks.pytorch.html) 
- [XGBoost](../api/mlrun.frameworks/mlrun.frameworks.xgboost.html) 
- [LightGBM](../api/mlrun.frameworks/mlrun.frameworks.lgbm.html) 
- [ONNX](../api/mlrun.frameworks/mlrun.frameworks.onnx.html)