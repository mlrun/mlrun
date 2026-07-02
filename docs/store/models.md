(models)=
# Model artifacts
You can work with models that are [stored locally](#locally-hosted-models) and [remotely](#remote-models).

**In this section**
- [Locally hosted models](#locally-hosted-models)
- [Train, exaluate, and log a model trained with scikit-learn](#train-exaluate-and-log-a-model-trained-with-scikit-learn)
- [Retrieve, run, and update a model using `models_path` and `test_set`](#retrieve-run-and-update-a-model-using-models_path-and-test_set)
- [Remote models](#remote-models)

**See also**
- {ref}`artifacts`

## Locally hosted models

An essential piece of artifact management and versioning is storing a model version. This allows you to experiment with different models and compare their performance, without having to worry about losing their previous results.

The simplest way to store a model named `my_model` is with the following code:

``` python
from pickle import dumps

model_data = dumps(model)
context.log_model(key="my_model", body=model_data, model_file="my_model.pkl")
```

You can also store any related metrics by providing a dictionary in the `metrics` parameter, such as `metrics={'accuracy': 0.9}`. Furthermore, any additional data that you would like to store along with the model can be specified in the `extra_data` parameter. For example `extra_data={'confusion': confusion.target_path}`

A convenient utility method, `eval_model_v2`, which calculates mode metrics is available in `mlrun.utils`.

## Train, exaluate, and log a model trained with scikit-learn

See example below for a simple model trained using scikit-learn (normally, you would send the data as input to the function). The last 2 lines evaluate the model and log the model (using {py:meth}`~mlrun.projects.MlrunProject.log_model`).

``` python
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pickle import dumps

import mlrun.execution
import mlrun.mlutils


def train_iris(context: mlrun.execution.MLClientCtx):

    # Basic scikit-learn iris SVM model
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = linear_model.LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # Evaluate model results and get the evaluation metrics
    eval_metrics = eval_model_v2(context, X_test, y_test, model)

    # Log model
    context.log_model(
        "model",
        body=dumps(model),
        artifact_path=context.artifact_subpath("models"),
        extra_data=eval_metrics,
        model_file="model.pkl",
        metrics=context.results,
        labels={"class": "sklearn.linear_model.LogisticRegression"},
    )
```

Save the code above to `train_iris.py`. The following code loads the function and runs it as a job. See [Artifact path](../store/artifacts.md#artifact-path) to learn how to set the artifact path.

``` python
import mlrun

project = mlrun.get_or_create_project("myproj")

gen_func = project.set_function(
    name="train_iris",
    func="<path to train_iris.py>",
    handler="train_iris",
    kind="job",
    image="mlrun/mlrun",
)

train_iris_func = project.set_function(gen_func).apply(auto_mount())

train_iris = train_iris_func.run(
    name="train_iris", handler="train_iris", output_path=output_path
)
```

## Retrieve, run, and update a model using `models_path` and `test_set`
You can now use `get_model` to read the model and run it. This function gets the model file, metadata, and extra data. The input can be either the path of the model, or the directory where the model resides. If you provide a directory, the function searches for the model file (by default it searches for `.pkl` files).

The following example gets the model from `models_path` and gets test data in `test_set` with the expected label provided as a column of the test data. The name of the column containing the expected label is provided in `label_column`. The example then retrieves the models, runs the model with the test data and updates the model with the metrics and results of the test data.

``` python
from pickle import load

import mlrun.execution
import mlrun.datastore
import mlrun.artifacts
import mlrun.mlutils


def test_model(
    context: MLClientCtx, models_path: DataItem, test_set: DataItem, label_column: str
):

    if models_path is None:
        models_path = context.artifact_subpath("models")
    xtest = test_set.as_df()
    ytest = xtest.pop(label_column)

    model_file, model_obj, _ = get_model(models_path)
    model = load(open(model_file, "rb"))

    extra_data = eval_model_v2(context, xtest, ytest.values, model)
    update_model(
        model_artifact=model_obj,
        extra_data=extra_data,
        metrics=context.results,
        key_prefix="validation-",
    )
```

To run the code, place the code above in `test_model.py` and use the following snippet. The model from the previous step is provided as the `models_path`:

``` python
import mlrun.platforms

gen_func = project.set_function(
    name="test_model",
    func="<path to test_model.py>",
    handler="test_model",
    kind="job",
    image="mlrun/mlrun",
)

func = project.set_function(gen_func).apply(auto_mount())

run = func.run(
    name="test_model",
    handler="test_model",
    params={"label_column": "label"},
    inputs={
        "models_path": train_iris.outputs["model"],
        "test_set": "https://s3.wasabisys.com/iguazio/data/iris/iris_dataset.csv",
    },
    output_path=output_path,
)
```

## Remote models
You can use models stored in a remote source, for example OpenAI. You can load the model from the remote source, without
saving it in your datastore. A remote {py:class}`~mlrun.artifacts.model.ModelArtifact` instance does not have any extra
data or similar facilities that the locally stored model artifact supports. Remote models are specified by the `ModelArtifact`
parameter `model_url`, which accepts various path schemas, as described in the next sections. You can specify the default
configuration for building clients with the `default_config` parameter.

### Guidelines for the parameter `model_url`:
- Remote model artifacts cannot be uploaded or downloaded. Consequently, the `upload` parameter cannot be set to `True`.
- The `model_dir` and `model_file` parameters cannot be specified.
- The `body` parameter cannot be specified.

### Credentials

For models not using a datastore profile, the MLRun code attempts to retrieve credentials from the environment (using `get_secret_or_env`).
For each type of schema, a standard secret name must be provided. For example, `OPENAI_API_KEY` for OpenAI, `HF_TOKEN` for HF, etc.

### HuggingFace
`huggingface://<model-path>`: Use the Hugging Face pipeline as a client to download models and use them. The URL contains the vendor and name of model, for example: `huggingface://google/gemma-3-27b-it`. Hugging Face secrets and parameters are:
- task / "HF_TASK", "text-generation" by default
- token / "HF_TOKEN"
- device / "HF_DEVICE"
- device_map/ "HF_DEVICE_MAP"
- trust_remote_code / "HF_TRUST_REMOTE_CODE"
- model_kwargs / "HF_MODEL_KWARGS"

### OpenAI
 `openai://<model-name>`: work with a model that supports the OpenAI protocol. Models are assumed to be models served by OpenAI. OpenAI secrets and parameters are:
- api_key / "OPENAI_API_KEY"
- endpoint / "OPENAI_BASE_URL"
- organization / "OPENAI_ORG_ID"
- project / "OPENAI_PROJECT_ID"
- timeout / "OPENAI_TIMEOUT"
- max_retries / "OPENAI_MAX_RETRIES"

### ds
`ds://<profile name>/<model-name>`: Use a datastore profile for model connection parameters. The profile must contain the required connection parameters: secrets and credentials, as well as parameters that determine the routing to the model (such as the endpoint URL), but not the actual model name. Since the profile does not contain the model name, it can be used for multiple models. 