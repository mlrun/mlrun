(working-with-data-and-model-artifacts)=
# Working with data and model artifacts

When running a training job, you need to pass in the data used for training, and save the resulting model. Both the data and model can be considered {ref}`artifacts <Artifacts>` in MLRun. In the context of an ML pipeline, the data is an `input` and the model is an `output`.

Consider the following snippet from a pipeline in the [Build and run automated ML pipelines and CI/CD](../tutorial/04-pipeline.html#build-and-run-automated-ml-pipelines-and-ci-cd) section of the docs:

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

This snippet trains a model using the data provided into `inputs` and passes the model to the rest of the pipeline using the `outputs`.

## Input data

The `inputs` parameter is a dictionary of key-value mappings. In this case, the input is the `dataset` (which is actually an output from a previous step). Within the training job, you can access the `dataset` input as an MLRun {ref}`data-items` (essentially a smart data pointer that provides convenience methods).

For example, this Python training function is expecting a parameter called `dataset` that is of type `DataItem`. Within the function, you can get the training set as a Pandas dataframe via the following:
```python
import mlrun

def train(context: mlrun.MLClientCtx, dataset: mlrun.DataItem, ...):
    df = dataset.as_df()
```
Notice how this maps to the parameter `datasets` that you passed into your `inputs`.

## Output model

The `outputs` parameter is a list of artifacts that were logged during the job. In this case, it is your newly trained `model`, however it could also be a dataset or plot. These artifacts are logged using the experiment tracking hooks via the MLRun execution context.

One way to log models is via MLRun auto-logging with {ref}`apply_mlrun <auto-logging-mlops>`. This saves the model, test sets, visualizations, and more as outputs. Additionally, you can use manual hooks to save datasets and models. For example, this Python training function uses both auto logging and manual logging:
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

Once your artifact is logged, it can be accessed throughout the rest of the pipeline. For example, for the pipeline snippet from the [Build and run automated ML pipelines and CI/CD](../tutorial/04-pipeline.html#build-and-run-automated-ml-pipelines-and-ci-cd) section of the docs, you can access your model like the following:
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

Notice how this maps to the parameter `model` that you passed into your `outputs`.


### Model artifacts
<!-- exists in standalone file in store -->

By storing multiple models, you can experiment with them,  
and compare their performance, without having to worry about losing the previous results.

The simplest way to store a model named `my_model` is with the following code:

``` python
from pickle import dumps
model_data = dumps(model)
context.log_model(key='my_model', body=model_data, model_file='my_model.pkl')
```

You can also store any related metrics by providing a dictionary in the `metrics` parameter, such as `metrics={'accuracy': 0.9}`. 
Furthermore, any additional data that you would like to store along with the model can be specified in the `extra_data` parameter. For example `extra_data={'confusion': confusion.target_path}`

A convenient utility method, `eval_model_v2`, which calculates mode metrics is available in `mlrun.utils`.

See example below for a simple model trained using scikit-learn (normally, you would send the data as input to the function). The last two 
lines evaluate the model and log the model.

``` python
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pickle import dumps

from mlrun.execution import MLClientCtx
from mlrun.mlutils import eval_model_v2

def train_iris(context: MLClientCtx):

    # Basic scikit-learn iris SVM model
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = linear_model.LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    
    # Evaluate model results and get the evaluation metrics
    eval_metrics = eval_model_v2(context, X_test, y_test, model)
    
    # Log model
    context.log_model("model",
                      body=dumps(model),
                      artifact_path=context.artifact_subpath("models"),
                      extra_data=eval_metrics, 
                      model_file="model.pkl",
                      metrics=context.results,
                      labels={"class": "sklearn.linear_model.LogisticRegression"})
```

Save the code above to `train_iris.py`. The following code loads the function and runs it as a job. See the [Quick start tutorial](../tutorial/01-mlrun-basics.html) to learn how to create the project and set the artifact path. 

``` python
from mlrun import code_to_function

gen_func = code_to_function(name='train_iris',
                            filename='train_iris.py',
                            handler='train_iris',
                            kind='job',
                            image='mlrun/ml-models')

train_iris_func = project.set_function(gen_func).apply(auto_mount())

train_iris = train_iris_func.run(name='train_iris',
                                 handler='train_iris',
                                 artifact_path=artifact_path)
```

You can now use `get_model` to read the model and run it. This function gets the model file, metadata, and extra data. The input can be 
either the path of the model, or the directory where the model resides. If you provide a directory, the function searches for the model file 
(by default it searches for .pkl files)

The following example gets the model from `models_path` and test data in `test_set` with the expected label provided as a column of the test 
data. The name of the column containing the expected label is provided in `label_column`. The example then retrieves the models, runs the 
model with the test data and updates the model with the metrics and results of the test data.

``` python
from pickle import load

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import get_model, update_model
from mlrun.mlutils import eval_model_v2

def test_model(context: MLClientCtx,
               models_path: DataItem,
               test_set: DataItem,
               label_column: str):

    if models_path is None:
        models_path = context.artifact_subpath("models")
    xtest = test_set.as_df()
    ytest = xtest.pop(label_column)

    model_file, model_obj, _ = get_model(models_path)
    model = load(open(model_file, 'rb'))

    extra_data = eval_model_v2(context, xtest, ytest.values, model)
    update_model(model_artifact=model_obj, extra_data=extra_data, 
                 metrics=context.results, key_prefix='validation-')
```

To run the code, place the code above in `test_model.py` and use the following snippet. The model from the previous step is provided as the `models_path`:

``` python
from mlrun.platforms import auto_mount
gen_func = code_to_function(name='test_model',
                            filename='test_model.py',
                            handler='test_model',
                            kind='job',
                            image='mlrun/ml-models')

func = project.set_function(gen_func).apply(auto_mount())

run = func.run(name='test_model',
                handler='test_model',
                params={'label_column': 'label'},
                inputs={'models_path': train_iris.outputs['model'],
                        'test_set': 'https://s3.wasabisys.com/iguazio/data/iris/iris_dataset.csv'}),
                artifact_path=artifact_path)
```

### Plot artifacts

Storing plots is useful to visualize the data and to show any information regarding the model performance. For example, you can store 
scatter plots, histograms and cross-correlation of the data, and for the model store the ROC curve and confusion matrix.

The following code creates a confusion matrix plot using [sklearn.metrics.plot_confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html#sklearn.metrics.plot_confusion_matrix) 
and stores the plot in the artifact repository:

``` python
from mlrun.artifacts import PlotArtifact
from mlrun.mlutils import gcf_clear

gcf_clear(plt)
confusion_matrix = metrics.plot_confusion_matrix(model,
                                                 xtest,
                                                 ytest,
                                                 normalize='all',
                                                 values_format = '.2g',
                                                 cmap=plt.cm.Blues)
confusion_matrix = context.log_artifact(PlotArtifact('confusion-matrix', body=confusion_matrix.figure_), 
                                        local_path='plots/confusion_matrix.html')
```

You can use the `update_dataset_meta` function to associate the plot with the dataset by assigning the value of the `extra_data` parameter:

``` python
from mlrun.artifacts import update_dataset_meta

extra_data = {'confusion_matrix': confusion_matrix}
update_dataset_meta(dataset, extra_data=extra_data)
```