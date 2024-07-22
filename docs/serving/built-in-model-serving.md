(using_built_in_model_serving_classes)=
# Using built-in model serving classes

MLRun includes built-in classes for commonly used frameworks. While you can {ref}`create your own class <custom-model-serving-class>`, 
it is often not necessary to write one if you use these standard classes.

The following table specifies, for each framework, the corresponding MLRun `ModelServer` serving class and its dependencies:

|framework       |serving class                               |dependencies    |
|:---------------|:-------------------------------------------|:---------------|
|Scikit-learn    |mlrun.frameworks.sklearn.SklearnModelServer |scikit-learn    |
|TensorFlow.Keras|mlrun.frameworks.tf_keras.TFKerasModelServer|tensorflow      |
|ONNX            |mlrun.frameworks.onnx.ONNXModelServer       |onnxruntime     |
|XGBoost         |mlrun.frameworks.xgboost.XGBoostModelServer |xgboost         |
|LightGBM        |mlrun.frameworks.lgbm.LGBMModelServer       |lightgbm        |
|PyTorch         |mlrun.frameworks.pytorch.PyTorchModelServer |torch           |

For GPU support, use the `mlrun/mlrun-gpu` image (adding GPU drivers and support).

## Example

The following code shows how to create a basic serving model using Scikit-learn.

``` python
import os
import urllib.request
import mlrun

model_path = os.path.abspath("sklearn.pkl")

# Download the model file locally
urllib.request.urlretrieve(
    mlrun.get_sample_path("models/serving/sklearn.pkl"), model_path
)

# Set the base project name
project_name_base = "serving-test"

# Initialize the MLRun project object
project = mlrun.get_or_create_project(
    project_name_base, context="./", user_project=True
)

serving_function_image = "mlrun/mlrun"
serving_model_class_name = "mlrun.frameworks.sklearn.SklearnModelServer"

# Create a serving function
serving_fn = mlrun.new_function(
    "serving", project=project.name, kind="serving", image=serving_function_image
)

# Add a model, the model key can be anything we choose. The class will be the built-in scikit-learn model server class
model_key = "scikit-learn"
serving_fn.add_model(
    key=model_key, model_path=model_path, class_name=serving_model_class_name
)
```

After the serving function is created, you can test it:

``` python
# Test data to send
my_data = {"inputs": [[5.1, 3.5, 1.4, 0.2], [7.7, 3.8, 6.7, 2.2]]}

# Create a mock server in order to test the model
mock_server = serving_fn.to_mock_server()

# Test the serving function
mock_server.test(f"/v2/models/{model_key}/infer", body=my_data)
```

Similarly, you can deploy the serving function and test it with some data:

``` python
# Deploy the serving function
serving_fn.apply(mlrun.auto_mount()).deploy()

# Check the result using the deployed serving function
serving_fn.invoke(path=f"/v2/models/{model_key}/infer", body=my_data)
```