(using-mlrun)=
# Using MLRun

MLRun is an open source AI orchestration platform for quickly building and managing continuous (gen) AI applications across their lifecycle. MLRun integrates into your development and CI/CD environment and automates the delivery of production data, ML pipelines, and online applications. MLRun significantly reduces engineering efforts, time to production, and computation resources.
With MLRun, you can choose any IDE on your local machine or on the cloud. MLRun breaks the silos between data, ML, software, and DevOps/MLOps teams, enabling collaboration and fast continuous improvements.

Get started with MLRun **{ref}`Tutorials and examples <tutorial>`**, **{ref}`Installation and setup guide <install-setup-guide>`**, 


This page explains how MLRun addresses the [**gen AI tasks**](#genai-tasks), [**MLOps tasks**](#mlops-tasks), and presents the [**MLRun core components**](#core-components).

See the supported data stores, development tools, services, platforms, etc., supported by MLRun's open architecture in **{ref}`ecosystem`**.


## Gen AI tasks

`````{div} 

````{grid} 4 
:gutter: 2


```{grid-item-card} Data management
:text-align: center
:link: .html
```

```{grid-item-card} Development 
:text-align: center
:link: .html
```

```{grid-item-card} Deployment
:text-align: center
:link: .html
```

```{grid-item-card} Live Ops
:text-align: center
:link: .html
```

````

`````


The [**gen AI development workflow**](./genai/genai-flow.html) section describes the different tasks and stages in detail.

### Data management


MLRun supports batch or realtime data processing at scale, data lineage and versioning, structured and unstructured data, and more. 
Removing inappropriate data at an early stage saves resources that would otherwise be required later on.


`````{div} full-width
{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Using LLMs to process unstructured data <./genai/data-mgmt/unstructured-data.html>`
{bdg-link-info}`Vector databases <./genai/data-mgmt/vector-databases.html>`
{bdg-link-info}`Guardrails for data management<./genai/data-mgmt/guardrails-data.html>`
<br> {octicon}`code-square` **Demo:**
{bdg-link-success}`Call center demo <https://github.com/mlrun/demo-call-center>`
<br> {octicon}`video` **Video:**
{bdg-link-warning}`Call center <https://youtu.be/YycMbxRgLBA>`
`````


### Development
Use MLRun to build an automated ML pipeline to: collect data, 
preprocess (prepare) the data, run the training pipeline, and evaluate the model.


`````{div} full-width
{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Working with RAG <./genai/development/working-with-rag.html>`
{bdg-link-info}`Evalating LLMs <./genai/development/evaluating-llms.html>`
{bdg-link-info}`Fine tuning LLMS <./genai/development/fine-tuning-llms.html>`
<br> {octicon}`code-square` **Demos:**
{bdg-link-success}`Call center demo <https://github.com/mlrun/demo-call-center>`
{bdg-link-success}`Build & deploy custom (fine-tuned) LLM models and applications <https://github.com/mlrun/demo-llm-tuning/blob/main>`
{bdg-link-success}`Interactive bot demo using LLMs <https://github.com/mlrun/demo-llm-bot/blob/main>`
<br> {octicon}`video` **Video:**
{bdg-link-warning}`Call center <https://youtu.be/YycMbxRgLBA>`
`````

### Deployment
MLRun serving can productize the newly trained LLM as a serverless function using real-time auto-scaling Nuclio serverless functions. 
The application pipeline includes all the steps from accepting events or data, contextualizing it with a state  preparing the required model features, 
inferring results using one or more models, and driving actions. 


`````{div} full-width
{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Serving gen AI models <./genai/deployment/genai_serving.html>`
{bdg-link-info}`GPU utilization <./genai/deployment/gpu_utilization.html>`
{bdg-link-info}`Gen AI realtime serving graph <./genai/deployment/genai_serving_graph.html>`
<br> {octicon}`code-square` **Tutorial:**
{bdg-link-primary}`Deploy LLM using MLRun <./tutorials/genai_01_basic_tutorial.html>`
<br> {octicon}`code-square` **Demos:**
{bdg-link-success}`Call center demo <https://github.com/mlrun/demo-call-center>`
{bdg-link-success}`Build & deploy custom (fine-tuned) LLM models and applications <https://github.com/mlrun/demo-llm-tuning/blob/main>`
{bdg-link-success}`Interactive bot demo using LLMs <https://github.com/mlrun/demo-llm-bot/blob/main>`
<br> {octicon}`video` **Video:**
{bdg-link-warning}`Call center <https://youtu.be/YycMbxRgLBA>`
`````

### Live Ops
Monitor all resources, data, model and application metrics to ensure performance. Then identify risks, control costs, and measure business KPIs.
Collect production data, metadata, and metrics to tune the model and application further, and to enable governance and explainability.


`````{div} full-width
{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Model monitoring <monitoring>`
{bdg-link-info}`Alerts and notifications <alerts-notifications>`
<br> {octicon}`code-square` **Tutorials:**
{bdg-link-primary}`Deploy LLM using MLRun <./tutorials/genai_01_basic_tutorial.html>`
<br> {octicon}`code-square` **Demo:**
{bdg-link-primary}`Build & deploy custom (fine-tuned) LLM models and applications <https://github.com/mlrun/demo-llm-tuning/blob/main>`
`````

## MLOps tasks

`````{div} 

````{grid} 4 
:gutter: 2

```{grid-item-card} Project management and CI/CD automation
:columns: 12
:text-align: center
:link: ./projects/project.html
```

```{grid-item-card} Ingest and process data
:columns: 4
:text-align: center
:link: ./data-prep/index.html
```

```{grid-item-card} Develop and train models 
:columns: 4
:text-align: center
:link: ./development/index.html
```

```{grid-item-card} Deploy models and app
:columns: 4
:text-align: center
:link: ./deployment/index.html
```

```{grid-item-card} Model Monitoring
:columns: 4
:text-align: center
:link: ./monitoring/index.html
```

```{grid-item-card} Alerts and Notifications
:columns: 4
:text-align: center
:link: ./concepts/alerts-notifications.html
```

````

`````

The [**MLOps development workflow**](./mlops-dev-flow.html) section describes the different tasks and stages in detail.
MLRun can be used to automate and orchestrate all the different tasks or just specific tasks (and integrate them with what you have already deployed).

### Project management and CI/CD automation

In MLRun the assets, metadata, and services (data, functions, jobs, artifacts, models, secrets, etc.) are organized into projects.
Projects can be imported/exported as a whole, mapped to git repositories or IDE projects (in PyCharm, VSCode, etc.), which enables versioning, collaboration, and CI/CD. 
Project access can be restricted to a set of users and roles.
{bdg-link-primary-line}`more... <./projects/project.html>`

`````{div} full-width
{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Projects and automation <./projects/ci-cd-automate.html>`
{bdg-link-info}`CI/CD integration <./projects/ci-integration.html>`
<br> {octicon}`code-square` **Tutorials:**
{bdg-link-primary}`Quick start <./tutorials/01-mlrun-basics.html>`
{bdg-link-primary}`Automated ML pipeline <./tutorials/04-pipeline.html>`
<br> {octicon}`video` **Videos:**
{bdg-link-warning}`Quick start <https://youtu.be/xI8KVGLlj7Q>`
`````



### Ingest and process data

MLRun provides abstract interfaces to various offline and online [**data sources**](./store/datastore.html), supports batch or realtime data processing at scale, data lineage and versioning, structured and unstructured data, and more. 
In addition, the MLRun [**Feature store**](./feature-store/feature-store.html) automates the collection, transformation, storage, catalog, serving, and monitoring of data features across the ML lifecycle and enables feature reuse and sharing.
{bdg-link-primary-line}`more... <./data-prep/index.html>`

`````{div} full-width
{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Ingest and process data <data-prep/index.html>`
{bdg-link-info}`Feature store <./feature-store/feature-store.html>`
{bdg-link-info}`Data and artifacts <./concepts/data.html>`
<br> {octicon}`code-square` **Tutorials:**
{bdg-link-primary}`quick start <./tutorials/01-mlrun-basics.html>`
{bdg-link-primary}`Feature store <./feature-store/basic-demo.html>`

`````

### Develop and train models

MLRun allows you to easily build ML pipelines that take data from various sources or the Feature Store and process it, 
train models at scale with multiple parameters, test models, track each experiment, and register, version and deploy 
models, etc. MLRun provides scalable built-in or custom model training services that integrate with any framework and 
can work with 3rd party training/auto-ML services. You can also bring your own pre-trained model and use it in the pipeline.

{bdg-link-primary-line}`more... <./development/index.html>`

`````{div} full-width
{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Develop and train models <development>`
{bdg-link-info}`Model training and tracking <./development/model-training-tracking.html>`
{bdg-link-info}`Batch runs and workflows <./concepts/runs-workflows.html>`
<br> {octicon}`code-square` **Tutorials:**
{bdg-link-primary}`Train, compare, and register models <./tutorials/02-model-training.html>`
{bdg-link-primary}`Automated ML pipeline <./tutorials/04-pipeline.html>`
<br> {octicon}`video` **Videos:**
{bdg-link-warning}`Train and compare models <https://youtu.be/bZgBsmLMdQo>`
`````

### Deploy models and applications


MLRun rapidly deploys and manages production-grade real-time or batch application pipelines using elastic and resilient serverless functions. MLRun addresses the entire ML application: intercepting application/user requests, running data processing tasks, inferencing using one or more models, driving actions, and integrating with the application logic.
{bdg-link-primary-line}`more... <./deployment/index.html>`

`````{div} full-width
{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Deploy models and applications <deployment>`
{bdg-link-info}`Realtime pipelines <./serving/serving-graph.html>`
{bdg-link-info}`Batch inference <./deployment/batch_inference.html>`
<br> {octicon}`code-square` **Tutorials:**
{bdg-link-primary}`Realtime serving <./tutorials/03-model-serving.html>`
{bdg-link-primary}`Batch inference <./tutorials/07-batch-infer.html>`
{bdg-link-primary}`Advanced pipeline <./tutorials/07-batch-infer.html>`
<br> {octicon}`video` **Videos:**
{bdg-link-warning}`Serve pre-trained models <https://youtu.be/OUjOus4dZfw>`
`````

### Monitor and alert

Observability is built into the different MLRun objects (data, functions, jobs, models, pipelines, etc.), eliminating the need for complex integrations and code instrumentation. With MLRun, you can observe the application/model resource usage and model behavior (drift, performance, etc.), define custom app metrics, and trigger alerts or retraining jobs.
{bdg-link-primary-line}`more... <./monitoring/index.html>`

`````{div} full-width
{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Model monitoring <monitoring>`
{bdg-link-info}`Alerts and notifications <alerts-notifications>`
<br> {octicon}`code-square` **Tutorials:**
{bdg-link-primary}`Model monitoring and drift detection <./tutorials/05-model-monitoring.html>`
`````

<a id="core-components"></a>
## MLRun core components

MLRun includes the following major components:

````{grid} 6
:gutter: 2

```{grid-item-card} Project management & automation (SDK, API, etc.)
:columns: 12
:text-align: center
:link: ./projects/project.html
```

```{grid-item-card} Serverless functions
:columns: 6 4 4 2
:text-align: center
:link: ./runtimes/functions.html
```

```{grid-item-card} Data & artifacts
:columns: 6 4 4 2
:text-align: center
:link: ./concepts/data.html
```

```{grid-item-card} Feature store
:columns: 6 4 4 2
:text-align: center
:link: ./feature-store/feature-store.html
```

```{grid-item-card} Batch runs & workflows 
:columns: 6 4 4 2
:text-align: center
:link: ./concepts/runs-workflows.html
```

```{grid-item-card} Real-time pipelines
:columns: 6 4 4 2
:text-align: center
:link: ./serving/serving-graph.html
```

```{grid-item-card} Monitoring
:columns: 6 4 4 2
:text-align: center
:link: ./monitoring/index.html
```

````

**{ref}`Project management <projects>`:** A service (API, SDK, DB, UI) that manages the different project assets (data, functions, jobs, workflows, secrets, etc.) and provides central control and metadata layer.  

**{ref}`Serverless functions <Functions>`:** An automatically deployed software package with one or more methods and runtime-specific attributes (such as image, libraries, command, arguments, resources, etc.).

**{ref}`Data & artifacts <data-feature-store>`:** Glueless connectivity to various data sources, metadata management, catalog, and versioning for structured/unstructured artifacts.

**{ref}`Feature store <feature-store>`:** Automatically collects, prepares, catalogs, and serves production data features for development (offline) and real-time (online) deployment using minimal engineering effort.

**{ref}`Batch Runs & workflows <workflows>`:** Execute one or more functions with specific parameters and collect, track, and compare all their results and artifacts.

**{ref}`Real-time serving pipeline <serving-graph>`:** Rapid deployment of scalable data and ML pipelines using real-time serverless technology, including API handling, data preparation/enrichment, model serving, ensembles, driving and measuring actions, etc.

**{ref}`Real-time monitoring <model-monitoring-overview>`:** Monitors data, models, resources, and production components and provides a feedback loop for exploring production data, identifying drift, alerting on anomalies or data quality issues, triggering retraining jobs, measuring business impact, etc.

**{ref}`Alerts and notifications <alerts-notifications>`:** Use alerts to identify and inform you of possible problem situations. Use notifications to report status on runs and pipelines.


```{toctree}
:maxdepth: 1
:hidden:
architecture
ecosystem
```