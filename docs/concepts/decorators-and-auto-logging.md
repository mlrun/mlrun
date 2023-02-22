(decorators-and-auto-logging)=
# Decorators and auto-logging

While it is possible to log results and artifacts using {ref}`the MLRun execution context<mlrun-execution-context>`, it is often more convenient to use the {py:func}`mlrun.handler` decorator.

## Basic example

Assume you have the following code in `train.py`

``` python
import pandas as pd
from sklearn.svm import SVC

def train_and_predict(train_data,
                      predict_input,
                      label_column='label'):

    x = train_data.drop(label_column, axis=1)
    y = train_data[label_column]

    clf = SVC()
    clf.fit(x, y)

    return list(clf.predict(predict_input))
```

With the `mlrun.handler` the python function itself would not change, and logging of the inputs and outputs would be automatic. The resultant code is as follows:

``` python
import pandas as pd
from sklearn.svm import SVC
import mlrun

@mlrun.handler(labels={'framework':'scikit-learn'},
               outputs=['prediction:dataset'],
               inputs={"train_data": pd.DataFrame,
                       "predict_input": pd.DataFrame})
def train_and_predict(train_data,
                      predict_input,
                      label_column='label'):

    x = train_data.drop(label_column, axis=1)
    y = train_data[label_column]

    clf = SVC()
    clf.fit(x, y)

    return list(clf.predict(predict_input))
```

To run the code, use the following example:

``` python
import mlrun
project = mlrun.get_or_create_project("mlrun-example", context="./", user_project=True)

trainer = project.set_function("train.py", name="train_and_predict", kind="job", image="mlrun/mlrun", handler="train_and_predict")

trainer_run = project.run_function(
    "train_and_predict", 
    inputs={"train_data": mlrun.get_sample_path('data/iris/iris_dataset.csv'),
            "predict_input": mlrun.get_sample_path('data/iris/iris_to_predict.csv')
           }
)
```

The outcome is a run with:
1. A label with key "framework" and value "scikit-learn".
2. Two inputs "train_data" and "predict_input" created from Pandas DataFrame.
3. An artifact called "prediction" of type "dataset". The contents of the dataset will be the return value (in this case the prediction result).

## Labels

The decorator gives you the option to set labels for the run. The `labels` parameter is a dictionary with keys and values to set for the labels.

## Input type parsing

The `mlrun.handler` decorator can also parse the input types, if they are specified. An equivalent definition is as follows:

``` python
@mlrun.handler(labels={'framework':'scikit-learn'},
               outputs=['prediction:dataset'])
def train_and_predict(train_data: pd.DataFrame,
                      predict_input: pd.DataFrame,
                      label_column='label'):

...
```

**Notice**: Type hints from the `typing` module (e.g. `typing.Optional`, `typing.Union`, `typing.List` etc.) are 
currently not supported but will be in the future.

> **Note:** If the inputs does not have a type input, the decorator assumes the parameter type in {py:class}`mlrun.datastore.DataItem`. If you specify `inputs=False`, all the run inputs are assumed to be of type `mlrun.datastore.DataItem`. You also have the option to specify a dictionary where each key is the name of the input and the value is the type.

## Logging return values as artifacts

If you specify the `outputs` parameter, the return values will be logged as the run artifacts. `outputs` expects a list; the length of the list must match the number of returned values.

The simplest option is to specify a list of strings. Each string contains the name of the artifact. You can also specify the artifact type by adding a colon after the artifact name followed by the type (`'name:artifact_type'`). The following are valid artifact types:

- dataset
- directory
- file
- object
- plot
- result

If you use only the name without the type, the following mapping is used:

| Python type              | Artifact type |
|--------------------------|---------------|
| pandas.DataFrame         | Dataset       |
| pandas.Series            | Dataset       |
| numpy.ndarray            | Dataset       |
| dict                     | Result        |
| list                     | Result        |
| tuple                    | Result        |
| str                      | Result        |
| int                      | Result        |
| float                    | Result        |
| bytes                    | Object        |
| bytearray                | Object        |
| matplotlib.pyplot.Figure | Plot          |
| plotly.graph_objs.Figure | Plot          |
| bokeh.plotting.Figure    | Plot          |


Refer to the {py:func}`mlrun.handler` for more details.

