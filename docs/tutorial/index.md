(tutorial)=
# Tutorials and Examples

The following tutorials provide a hands-on introduction to using MLRun to implement a data science workflow and automate machine-learning operations (MLOps).

- [**Quick-start Tutorial**](./01-mlrun-basics.html) ({octicon}`video` [**watch video**](https://youtu.be/xI8KVGLlj7Q))
- [**Targeted Tutorials**](#other-tutorial)
- [**End to End Demos**](#e2e-demos)

(quick-start-tutorial)=

````{card} Make sure you start with the Quick start tutorial to understand the basics
```{button-link} ./01-mlrun-basics.html
:color: primary
:shadow:
:expand:
:click-parent:
Introduction to MLRun - Use serverless functions to train and deploy models
```
````

```{toctree}
:maxdepth: 1
:hidden:
01-mlrun-basics
02-model-training
03-model-serving
04-pipeline
05-model-monitoring
06-add-mlops-to-code
07-batch-infer
../feature-store/basic-demo
MLRun demos repository <https://github.com/mlrun/demos>
```

(other-tutorial)=
## Targeted Tutorials

Each of the following tutorials is a dedicated Jupyter notebook. You can download them by clicking the `download` icon at the top of each page.


`````{div} full-width

````{grid} 3
:gutter: 2

```{grid-item-card} Train, Compare, and Register Models
:link: ./02-model-training.html
Demo of training ML models, hyper-parameters, track and compare experiments, register and use the models.
```

```{grid-item-card} Serving Pre-trained ML/DL models
:link: ./03-model-serving.html
How to deploy real-time serving pipelines with MLRun Serving and different types of pre-trained ML/DL models.
```

```{grid-item-card} Projects & Automated ML Pipeline
:link: ./04-pipeline.html
How to work with projects, source control (git), CI/CD, to easily build and deploy multi-stage ML pipelines.
```

```{grid-item-card} Real-time Monitoring & Drift Detection
:link: ./05-model-monitoring.html
Demonstrate MLRun Serving pipelines, MLRun model monitoring, and automated drift detection  
```

```{grid-item-card} Add MLOps to existing code
:link: ./06-add-mlops-to-code.html
Turn a Kaggle research notebook to a production ML micro-service from with MLRun and minimal code changes.
```

```{grid-item-card} Basic Feature store example (stocks)
:link: ../feature-store/basic-demo.html
Understand MLRun feature store with a simple example, build, transform, and serve features in batch and in real-time.
```

```{grid-item-card} Batch Inference and Drift Detection
:link: ./07-batch-infer.html
Use MLRun batch inference function (from MLRun marketplace), run it as a batch job and generate drift reports.
```

```{grid-item-card} Advanced Real-Time Pipeline
:link: ../serving/graph-example.html
Demonstrate multi-step online pipeline with data prep, ensemble, model serving, post processing 
```

```{grid-item-card} Feature Store End-to-End Demo
:link: ../feature-store/end-to-end-demo/index.html
Use the feature store with data ingestion, model training, model serving and automated pipeline.
```

````
`````

(e2e-demos)=
## End to End Demos

You can find different end-to-end demos in MLRun demos repository at [**github.com/mlrun/demos**](https://github.com/mlrun/demos).

## Running the demos in Open Source MLRun

By default, these demos work with the online feature store, which is currently not part of the Open Source MLRun default deployment:
- fraud-prevention-feature-store 
- network-operations
- azureml_demo