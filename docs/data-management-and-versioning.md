# Artifact Management and versioning <!-- omit in toc -->

- [Overview](#overview)
- [Datasets](#datasets)
- [Plots](#plots)
- [Models](#models)

## Overview
An artifact is any data that is produced and/or consumed by functions or jobs.

The artifacts are stored in the project and are divided to 3 main types:
1. **Datasets** — any data , such as tables and DataFrames.
2. **Plots** — images, figures, and plotlines.
3. **Models** — all trained models.

From the projects page, click on the **Artifacts** link to view all the artifacts stored in the project
<br><br>
<img src="_static/images/project-artifacts.png" alt="projects-artifacts" width="800"/>

You can search the artifacts based on time and labels.
In the Monitor view, you can view per artifact its location, the artifact type, labels, the producer of the artifact, the artifact owner, last update date.

Per each artifact you can view its content as well as download the artifact.

## Datasets

Storing datasets is important in order to have a record of the data that was used to train the model, as well as storing any processed data. MLRun comes with built-in support for DataFrame format, and can not just store the DataFrame, but also provide the user information regarding the data, such as statistics.

The simplest way to store a dataset is with the following code:

``` python
context.log_dataset(key='my_data', df=df)
```

Where `key` is the the name of the artifact and `df` is the DataFrame. By default, MLRun will store a short preview of 20 lines. You can change the number of lines by using the `preview` parameter and setting it to a different value.

MLRun will also calculate statistics on the DataFrame on all numeric fields. You can enable statistics regardless to the DataFrame size by setting the `stats` parameter to `True`.

## Plots

Storing plots is useful to visualize the data and to show any information regarding the model performance. For example, one can store scatter plots,  histograms and cross-correlation of the data, and for the model store the ROC curve and confusion matrix.

## Models

An essential piece of artifact management and versioning is storing a model version. This allows the users to experiment with different models and compare their performance, without having to worry about losing their previous results.

The simplest way to store a model named `model` is with the following code:

``` python
from pickle import dumps
model_data = dumps(model)
context.log_model(key='my_model', body=model_data, model_file='my_model.pkl')
```

You can also store any related metrics by providing a dictionary in the `metrics` paramterer, such as `metrics={'accuracy': 0.9}`. Furthermore, any additional data that you would like to store along with the model can be specifieid in the `extra_data` parameter. For example `extra_data={'confusion': confusion.target_path}`