(architecture)=
# Using MLRun 

```{div} full-width
MLRun is an open MLOps platform for quickly building and managing continuous ML applications across their lifecycle. MLRun integrates into your development and CI/CD environment and automates the delivery of production data, ML pipelines, and online applications, significantly reducing engineering efforts, time to production, and computation resources. 
MLRun breaks the silos between data, ML, software, and DevOps/MLOps teams, enabling collaboration and fast continuous improvements.

Get started with MLRun [**Tutorials and Examples**](./tutorial/index.html), [**Installation and setup guide**](./install.html)
, or read about [**MLRun Architecture**](./architecture.html)
```

This page explains how MLRun addressed the [**MLOps Tasks**](#mlops-tasks) and the [**MLRun core components**](#core-components)

<a id="mlops-tasks"></a>
## MLOps Tasks

`````{div} full-width

````{grid} 4
:gutter: 2

```{grid-item-card} Project management and CI/CD automation
:columns: 12
:text-align: center
:link: ./projects/project.html
```

```{grid-item-card} Ingest and process data
:text-align: center
:link: ./data-prep/index.html
```

```{grid-item-card} Develop and train models 
:text-align: center
:link: ./development/index.html
```

```{grid-item-card} Deploy models and apps
:text-align: center
:link: ./deployment/index.html
```

```{grid-item-card} Monitor and alert
:text-align: center
:link: ./monitoring/index.html
```

````

The [**MLOps development workflow**](./mlops-dev-flow.html) section describes the different tasks and stages in detail.
MLRun can be used to automate and orchestrate all the different tasks or just specific tasks (and integrate them with what you have already deployed).

### Project management and CI/CD automation

In MLRun the assets, metadata, and services (data, functions, jobs, artifacts, models, secrets, etc.) are organized into projects.
Projects can be imported/exported as a whole, mapped to git repositories or IDE projects (in PyCharm, VSCode, etc.), which enables versioning, collaboration, and CI/CD. 
Project access can be restricted to a set of users and roles.
{bdg-link-primary-line}`more... <./projects/project.html>`

{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Projects and Automation <./projects/project.html>`
{bdg-link-info}`CI/CD Integration <./projects/ci-integration.html>`
, {octicon}`code-square` **Tutorials:**
{bdg-link-primary}`quick start <./tutorial/01-mlrun-basics.html>`
{bdg-link-primary}`Automated ML Pipeline <./tutorial/04-pipeline.html>`
, {octicon}`video` **Videos:**
{bdg-link-warning}`quick start <https://youtu.be/xI8KVGLlj7Q>`


### Ingest and process data

MLRun provides abstract interfaces to various offline and online [**data sources**](./concepts/data-feature-store.html), supports batch or realtime data processing at scale, data lineage and versioning, and more. 
In addition, the MLRun [**Feature Store**](./feature-store/feature-store.html) automates the collection, transformation, storage, catalog, serving, and monitoring of data features across the ML lifecycle and enables feature reuse and sharing.
{bdg-link-primary-line}`more... <./data-prep/index.html>`

{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Feature Store <./feature-store/feature-store.html>`
{bdg-link-info}`Data & Artifacts <./concepts/data-feature-store.html>`
, {octicon}`code-square` **Tutorials:**
{bdg-link-primary}`quick start <./tutorial/01-mlrun-basics.html>`
{bdg-link-primary}`Feature Store <./feature-store/basic-demo.html>`

### Develop and train models

MLRun allows you to easily build ML pipelines that take data from various sources or the Feature Store and process it, train models at scale with multiple parameters, test models, tracks each experiments, register, version and deploy models, etc. MLRun provides scalable built-in or custom model training services or can work with 3rd party training/auto-ML services.
{bdg-link-primary-line}`more... <./development/index.html>`

{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Model Training and Tracking <./development/model-training-tracking.html>`
{bdg-link-info}`Batch Runs and Workflows <./concepts/runs-workflows.html>`
, {octicon}`code-square` **Tutorials:**
{bdg-link-primary}`Train & Eval Models <./tutorial/02-model-training.html>`
{bdg-link-primary}`Automated ML Pipeline <./tutorial/04-pipeline.html>`

### Deploy models and applications

MLRun rapidly deploys and manages production-grade real-time or batch application pipelines using elastic and resilient serverless functions. MLRun addresses the entire ML application: intercepting application/user requests, running data processing tasks, inferencing using one or more models, driving actions, and integrating with the application logic.
{bdg-link-primary-line}`more... <./deployment/index.html>`

{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Realtime Pipelines <./serving/serving-graph.html>`
{bdg-link-info}`Batch Inference <./concepts/TBD.html>`
, {octicon}`code-square` **Tutorials:**
{bdg-link-primary}`Realtime Serving <./tutorial/03-model-serving.html>`
{bdg-link-primary}`Batch Inference <./tutorial/07-batch-infer.html>`
{bdg-link-primary}`Advanced Pipeline <./tutorial/07-batch-infer.html>`

### Monitor and alert

Observability is built into the different MLRun objects (data, functions, jobs, models, pipelines, etc.), eliminating the need for complex integrations and code instrumentation. With MLRun, you can observe the application/model resource usage and model behavior (drift, performance, etc.), define custom app metrics, and trigger alerts or retraining jobs.
{bdg-link-primary-line}`more... <./monitoring/index.html>`

{octicon}`mortar-board` **Docs:**
{bdg-link-info}`Model Monitoring Overview <./monitoring/model-monitoring-deployment.html>`
, {octicon}`code-square` **Tutorials:**
{bdg-link-primary}`Model Monitoring & Drift Detection <./tutorial/05-model-monitoring.html>`

`````

<a id="core-components"></a>
## MLRun Core components

MLRun includes the following major components

`````{div} full-width

````{grid} 6
:gutter: 2

```{grid-item-card} Project Management & Automation (SDK, API, etc.)
:columns: 12
:text-align: center
:link: ./projects/project.html
```

```{grid-item-card} Serverless Functions
:columns: 6 4 4 2
:text-align: center
:link: ./runtimes/functions.html
```

```{grid-item-card} Data & Artifacts
:columns: 6 4 4 2
:text-align: center
:link: ./concepts/data.html
```

```{grid-item-card} Feature Store
:columns: 6 4 4 2
:text-align: center
:link: ./feature-store/feature-store.html
```

```{grid-item-card} Batch Runs & Workflows 
:columns: 6 4 4 2
:text-align: center
:link: ./concepts/runs-workflows.html
```

```{grid-item-card} Real-time Pipelines
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

[**Project Management:**](./projects/project.html) A service (API, SDK, DB, UI) that manages the different project assets (data, functions, jobs, workflows, secrets, etc.) and provides central control and metadata layer.  

[**Serverless Functions:**](./runtimes/functions.html) automatically deployed software package with one or more methods and runtime-specific attributes (such as image, libraries, command, arguments, resources, etc.)

[**Data & Artifacts:**](./concepts/data-feature-store.html) Glueless connectivity to various data sources, metadata management, catalog, and versioning for structures/unstructured artifacts.

[**Feature Store:**](./feature-store/feature-store.html) automatically collects, prepares, catalogs, and serves production data features for development (offline) and real-time (online) deployment using minimal engineering effort.

[**Batch Runs & Workflows:**](./concepts/runs-workflows.html) Execute one or more functions with specific parameters and collect, track, and compare all their results and artifacts.

[**Real-Time Serving Pipeline:**](./serving/serving-graph.html) Rapid deployment of scalable data and ML pipelines using real-time serverless technology, including API handling, data preparation/enrichment, model serving, ensembles, driving and measuring actions, etc.

[**Real-Time monitoring:**](./monitoring/index.html) monitors data, models, resources, and production components and provides a feedback loop for exploring production data, identifying drift, alerting on anomalies or data quality issues, triggering retraining jobs, measuring business impact, etc.

`````

```{toctree}
:hidden:
:maxdepth: 1

architecture
```
