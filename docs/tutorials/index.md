(tutorials-all)=
# Tutorials and Examples

The following tutorials provide a hands-on introduction to using MLRun to implement a data science workflow and automate machine-learning operations (MLOps).

- [**Gen AI tutorials**](#gen-ai-tutorials)
- [**Machine learning tutorials**](#other-tutorial)
- [**End to End Demos**](#e2e-demos)

<iframe width="560" height="315" src="https://www.youtube.com/embed/xI8KVGLlj7Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe><br><br>

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
genai_01_basic_tutorial
genai-02-model-monitor-llm
ml-index
demos
/cheat-sheet
```

## Gen AI tutorials


`````{div}

````{grid} 3
:gutter: 2

```{grid-item-card} Deploy LLM using MLRun
:link: ./genai_01_basic_tutorial.html
How to copy a dataset into your cluster, deploy an LLM in the cluster, and run your function.
```
```{grid-item-card} Model monitoring using LLM
:link: ./genai-02-model-monitor-llm.html
Set up an effective model monitoring system that leverages LLMs to maintain high standards for deployed models.
```
```{grid-item-card} Model monitoring 
:link: ./05-model-monitoring.html
Demonstrate MLRun Serving pipelines, MLRun model monitoring, and automated drift detection.
```

````
`````


(other-tutorial)=
## Machine learning tutorials

Each of the following tutorials is a dedicated Jupyter notebook. You can download them by clicking the download icon <img src="../_static/images/icon-download.png">at the top of each page.


`````{div}

````{grid} 3
:gutter: 2

```{grid-item-card} Train, compare, and register Models
:link: ./02-model-training.html
Demo of training ML models, hyper-parameters, track and compare experiments, register and use the models.
```

```{grid-item-card} Serving pre-trained ML/DL models
:link: ./03-model-serving.html
How to deploy real-time serving pipelines with MLRun Serving and different types of pre-trained ML/DL models.
```

```{grid-item-card} Projects & automated ML pipeline
:link: ./04-pipeline.html
How to work with projects, source control (git), CI/CD, to easily build and deploy multi-stage ML pipelines.
```

```{grid-item-card} Real-time monitoring & drift detection
:link: ./05-model-monitoring.html
Demonstrate MLRun Serving pipelines, MLRun model monitoring, and automated drift detection.
```

```{grid-item-card} Add MLOps to existing code
:link: ./07-add-mlops-to-code.html
Turn a Kaggle research notebook to a production ML micro-service with minimal code changes using MLRun.
```

```{grid-item-card} Basic feature store example (stocks)
:link: ../feature-store/basic-demo.html
Understand MLRun feature store with a simple example: build, transform, and serve features in batch and in real-time.
```

```{grid-item-card} Batch inference and drift detection
:link: ./06-batch-infer.html
Use MLRun batch inference function (from MLRun Function Hub), run it as a batch job, and generate drift reports.
```

```{grid-item-card} Advanced real-time pipeline
:link: ../serving/graph-example.html
Demonstrates a multi-step online pipeline with data prep, ensemble, model serving, and post processing. 
```

```{grid-item-card} Feature store end-to-end demo
:link: ../feature-store/end-to-end-demo/index.html
Use the feature store with data ingestion, model training, model serving, and automated pipeline.
```

````
`````

(e2e-demos)=
## End to end demos

See {ref}`demos`.


## Cheat sheet

If you already know the basics, use the [cheat sheet](../cheat-sheet.html) as a guide to typical use cases and their flows/SDK.