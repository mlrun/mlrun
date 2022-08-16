(tutorial)=
# Tutorials and examples

The following tutorials provide a hands-on introduction to using MLRun to implement a data science workflow and automate machine-learning operations (MLOps).

Make sure you start with the [**Quick start tutorial**](./01-mlrun-basics.html) to understand the basics.

Each of the following tutorials is a dedicated Jupyter notebook. You can download them by clicking the `download` icon at the top of each page.

```{toctree}
:maxdepth: 1
:hidden:
01-mlrun-basics
02-model-training
03-model-serving
04-pipeline
./taxi/apply-mlrun-on-existing-code
../feature-store/basic-demo
../feature-store/end-to-end-demo/index

```

````{grid} 2
:gutter: 2
```{grid-item-card} Quick-start tutorial
:link: ./01-mlrun-basics.html

Introduction to MLRun - Use serverless functions to train and deploy models.
```

```{grid-item-card} Train, Compare, and Register Models
:link: ./02-model-training.html

Quick overview of training ML models using MLRun MLOps orchestration framework.
```

```{grid-item-card} Serving ML/DL models
:link: ./03-model-serving.html

How to serve standard ML/DL models using MLRun Serving.
```

```{grid-item-card} Projects and Automated ML Pipeline
:link: ./04-pipeline.html

How to work with projects, source control (git), and automating the ML pipeline.
```

```{grid-item-card} Feature store example (stocks)
:link: ../feature-store/basic-demo.html

Build features with complex transformations in batch and serve in real-time.
```

````

You can find different end-to-end demos in MLRun demos repository at [**github.com/mlrun/demos**](https://github.com/mlrun/demos).
<!-- Alternatively, use the interactive MLRun [**Katacoda Scenarios**](https://www.katacoda.com/mlrun) that teach how to install and use MLRun. -->See also:

```{toctree}
:maxdepth: 1

MLRun demos repository <https://github.com/mlrun/demos>

```
## Running the demos in Open Source MLRun

By default, these demos work with the online feature store, which is currently not part of the Open Source MLRun default deployment:
- fraud-prevention-feature-store 
- network-operations
- azureml_demo