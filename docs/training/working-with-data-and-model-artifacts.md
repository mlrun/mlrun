(working-with-data-and-model-artifacts)=
# Working with Data and Model Artifacts

When running a training job, you will need to pass in data to train and save the resulting model. Both the data and model can be considered [Artifacts](https://docs.mlrun.org/en/latest/store/artifacts.html) in MLRun. In the context of an ML pipeline, the data would be an `input` and the model would be an `output`.

Consider the following snippet from a pipeline in the [Build and run automated ML pipelines and CI/CD](https://docs.mlrun.org/en/latest/tutorial/04-pipeline.html#build-and-run-automated-ml-pipelines-and-ci-cd) section of the docs:

```python
# Ingest data
...

# Train a model using the auto_trainer hub function
train = mlrun.run_function(
    "hub://auto_trainer",
    inputs={"dataset": ingest.outputs["dataset"]},
    params = {
        "model_class": "sklearn.ensemble.RandomForestClassifier",
        "train_test_split_size": 0.2,
        "label_columns": "label",
        "model_name": 'cancer',
    }, 
    handler='train',
    outputs=["model"],
)

### Deploy model
...
```

This snippet will train a model using the data provided into `inputs` and pass the model to the rest of the pipeline using the `outputs`.

### Input Data

The `inputs` parameter is a dictionary of key-value mappings. In this case, our input is our `dataset` (which is actually an output from a previous step). Within our training job, we can access the `dataset` input as an MLRun [DataItem](https://docs.mlrun.org/en/latest/concepts/data-items.html) (essentially a smart data pointer that provides convenience methods).

For example, this Python training function is expecting a parameter called `dataset` that is of type `DataItem`. Within the function, we can get our training set as a Pandas dataframe via the following:
```python
import mlrun

def train(context: mlrun.MLClientCtx, dataset: mlrun.DataItem, ...):
    df = dataset.as_df()
```
Notice how this maps to the parameter `datasets` that we passed into our `inputs`.

### Output Model

The `outputs` parameter is a list of artifacts that were logged during the job. In this case, it is our newly trained `model`, however it could also be a dataset or plot. These artifacts are logged using the experiment tracking hooks via the MLRun execution context.

One way to log models is via MLRun [Auto Logging](https://docs.mlrun.org/en/latest/concepts/auto-logging-mlops.html). This will save the model, test set, visualizations, and more as outputs. Additionally, you can use manual hooks to save datasets and models. For example, this Python training function uses both auto logging and manual logging:
```python
import mlrun
from mlrun.frameworks.sklearn import apply_mlrun
from sklearn import ensemble
import cloudpickle

def train(context: mlrun.MLClientCtx, dataset: mlrun.DataItem, ...):
    # Prep data using df
    df = dataset.as_df()
    X_train, X_test, y_train, y_test = ...
    
    # Apply auto logging
    model = ensemble.GradientBoostingClassifier(...)
    apply_mlrun(model=model, model_name=model_name, x_test=X_test, y_test=y_test)

    # Train
    model.fit(X_train, y_train)
    
    # Manual logging
    context.log_dataset(key="X_test_dataset", df=X_test)
    context.log_model(key="my_model", body=cloudpickle.dumps(model), model_file="model.pkl")
```

Once your artifact is logged, it can be accessed throughout the rest of the pipeline. For example, for our pipeline snippet from the [Build and run automated ML pipelines and CI/CD](https://docs.mlrun.org/en/latest/tutorial/04-pipeline.html#build-and-run-automated-ml-pipelines-and-ci-cd) section of the docs, we can access our model like the following:
```python
# Train a model using the auto_trainer hub function
train = mlrun.run_function(
    "hub://auto_trainer",
    inputs={"dataset": ingest.outputs["dataset"]},
    ...
    outputs=["model"],
)

# Get trained model
model = train.outputs["model"]
```

Notice how this maps to the parameter `model` that we passed into our `outputs`.