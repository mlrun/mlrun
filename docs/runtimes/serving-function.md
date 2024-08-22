(serving-function)=
# Function of type `serving`

Deploying models in MLRun uses the function type {py:meth}`~mlrun.runtimes.ServingRuntime <serving>`. You can create a serving function using the `set_function()` call from a notebook. 
You can also import an existing serving function/template from the {ref}`load-from-hub`.

## Creating a basic serving model using Scikit-learn

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
project_name_base = "serving-project"

# Initialize the MLRun project object
project = mlrun.get_or_create_project(
    project_name_base, context="./", user_project=True
)

serving_function_image = "mlrun/mlrun"
serving_model_class_name = "mlrun.frameworks.sklearn.SklearnModelServer"

# Create a serving function within a project
serving_fn = mlrun.project.set_function(
    "serving", kind="serving", image=serving_function_image
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


## Using GIT with a serving function

This example illustrates how to use Git with serving function:

```python
project = mlrun.get_or_create_project("serving-git", "./")
project.set_source(
    source="git://github.com/<username>/<repo>.git#main", pull_at_runtime=True
)
function = project.set_function(
    name="serving",
    kind="serving",
    with_repo=True,
    func="<python-file>",
    image="mlrun/mlrun",
)
function.add_model("serve", "<model_path>", class_name="MyClass")
project.deploy_function(function="serving")
```



**See also**
- {ref}`Real-time serving pipelines (graphs) <serving-graph>`: higher level real-time graphs (DAG) over one or more Nuclio functions
- {ref}`Serving graphs demos and tutorials <demos-serving>` 
- {ref}`Real-time serving <mlrun-serving-overview>`
- {ref}`Serving pre-trained ML/DL models <serving-ml-dl-models>`

