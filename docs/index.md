(architecture)=
# Using MLRun 

```{div} full-width
MLRun is an open MLOps platform for quickly building and managing continuous ML applications across their lifecycle. MLRun integrates into your development and CI/CD environment and automates the delivery of production data, ML pipelines, and online applications, significantly reducing engineering efforts, time to production, and computation resources. 
MLRun breaks the silos between data, ML, software, and DevOps/MLOps teams, enabling collaboration and fast continuous improvements.
```

- [**MLOps Tasks**](#mlops-tasks)
- [**MLRun Core components**](#core-components), [**Architecture and Background**](#architecture)
- Get started with [**Tutorials and Examples**](./tutorial/index.html), and [**Installation and setup guide**](./install.html)

<a id="mlops-tasks"></a>
## MLOps Tasks

`````{div} full-width

````{grid} 4
:gutter: 2

```{grid-item-card} Project lifecycle, Metadata, and CI/CD Automation
:columns: 12
:text-align: center
:link: ./projects/project.html
```

```{grid-item-card} Data Ingestion and Processing
:text-align: center
:link: ./data-collect-prep/ingest-process-data.html
```

```{grid-item-card} Continious Model Development and Training 
:text-align: center
:link: ./model_development/continuous-model-development.html
```

```{grid-item-card} Deployment of Models and ML applications
:text-align: center
:link: ./serving/serving-overview.html
```

```{grid-item-card} Model, Data, and Application Monitoring
:text-align: center
:link: ./model_monitoring/monitoring.html
```

````

The [**MLOps development workflow**]() section describes the different tasks and stages in detail.
MLRun can be used to automate and orchestrate all the different tasks or just specific tasks (and integrate them with what you already have deployed).

### Project lifecycle, Metadata, and CI/CD Automation

In MLRun the assets and services (data, functions, jobs, artifacts, models, secrets, etc.) are organized into projects. Projects have access policies and can be imported/exported as a whole. 

Projects can be imported/exported as a whole, mapped to git repositories or IDE projects (in PyCharm, VSCode, etc.), which enables versioning, collaboration, and CI/CD. Project access can be restricted to a set of users and roles.
{bdg-link-primary-line}`more... <./projects/project.html>`

{octicon}`mortar-board` Docs:
{bdg-link-info}`Projects and Automation <./projects/project.html>`
{bdg-link-info}`CI/CD Integration <./projects/ci-integration.html>`
, {octicon}`code-square` Tutorials:
{bdg-link-primary}`quick start <./tutorial/01-mlrun-basics.html>`
{bdg-link-primary}`Automated ML Pipeline <./tutorial/04-pipeline.html>`
, {octicon}`video` Videos:
{bdg-link-warning}`quick start <https://youtu.be/xI8KVGLlj7Q>`


### Data Ingestion and Processing

MLRun provides abstract interfaces to various offline and online [**data sources**](./concepts/data-feature-store.html), supports batch or realtime data processing at scale, data lineage and versioning, and more. 
In addition, MLRun [**Feature Store**](./feature-store/feature-store.html) automates the collection, transformation, storage, catalog, serving, and monitoring of data features across the ML lifecycle and enables feature reuse and sharing.
{bdg-link-primary-line}`more... <./data-collect-prep/ingest-process-data.html>`

{octicon}`mortar-board` Docs:
{bdg-link-info}`Feature Store <./feature-store/feature-store.html>`
{bdg-link-info}`Data & Artifacts <./concepts/data-feature-store.html>`
, {octicon}`code-square` Tutorials:
{bdg-link-primary}`quick start <./tutorial/01-mlrun-basics.html>`
{bdg-link-primary}`Feature Store <./feature-store/basic-demo.html>`

### Continious Model Development and Training

MLRun allows you to build ML pipelines that take data from various sources or the Feature Store, process and validate the data, train the model at scale with multiple parameters, test the model, register in the model repository, etc. MLRun provides scalable built-in or custom model training services or can work with 3rd party training/auto-ML services.

MLRun orchestrates the execution of the pipelines, tracks the experiments and results, versions the data, and integrates natively with CI/CD frameworks.
{bdg-link-primary-line}`more... <./model_development/continuous-model-development.html>`

{octicon}`mortar-board` Docs:
{bdg-link-info}`Model Training and Tracking <./model-development/model-training-tracking.html>`
{bdg-link-info}`Batch Runs and Workflows <./concepts/runs-workflows.html>`
, {octicon}`code-square`
Tutorials:
{bdg-link-primary}`Train & Eval Models <./tutorial/02-model-training.html>`
{bdg-link-primary}`Automated ML Pipeline <./tutorial/04-pipeline.html>`

### Deployment of Models and ML applications

MLRun rapidly deploys and manages production-grade real-time or batch pipelines using elastic serverless functions. MLRun pipelines intercept application/user requests, run data pre-processing tasks, infer using one or more models, drive actions, and integrate with the application logic.
{bdg-link-primary-line}`more... <./serving/serving-overview.html>`

{octicon}`mortar-board` Docs:
{bdg-link-info}`Realtime Pipelines <./serving/serving-graph.html>`
{bdg-link-info}`Batch Inference <./concepts/TBD.html>`
, {octicon}`code-square` Tutorials:
{bdg-link-primary}`Realtime Serving <./tutorial/03-model-serving.html>`
{bdg-link-primary}`Batch Inference <./tutorial/07-batch-infer.html>`
{bdg-link-primary}`Advanced Pipeline <./tutorial/07-batch-infer.html>`

### Model, Data, and Application Monitoring

Observability is built into the different MLRun objects (data, functions, jobs, models, pipelines, etc.), eliminating the need for complex integrations and code instrumentation. With MlRun, you can observe the application/model resource usage and model behavior (drift, performance, etc.) and define custom app metrics.
{bdg-link-primary-line}`more... <./model_monitoring/index.html>`

{octicon}`mortar-board` Docs:
{bdg-link-info}`Model Monitoring Overview <./model_monitoring/model-monitoring-deployment.html>`
, {octicon}`code-square` Tutorials:
{bdg-link-primary}`Model Monitoring & Drift Detection <./tutorial/05-model-monitoring.html>`

`````


## MLRun Core components

MLRun includes the following major components (click the link to browse to the relevant section)

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
:link: ./concepts/data-feature-store.html
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
:link: ./model_monitoring/index.html
```

````

[**Project Management:**](./projects/project.html) A service (API, SDK, DB, UI) that manages the different project assets (data, functions, jobs, workflows, secrets, etc.) and provides central control and metadata layer.  

[**Serverless Functions:**](./runtimes/functions.html) automatically deployed software package with one or more methods and runtime-specific attributes (such as image, libraries, command, arguments, resources, etc.)

[**Data & Artifacts:**](./concepts/data-feature-store.html) Glueless connectivity to various data sources, metadata management, catalog, and versioning for structures/unstructured artifacts.

[**Feature Store:**](./feature-store/feature-store.html) automatically collects, prepares, catalogs, and serves production data features for development (offline) and real-time (online) deployment using minimal engineering effort.

[**Batch Runs & Workflows:**](./concepts/runs-workflows.html) Execute one or more functions with specific parameters and collect, track, and compare all their results and artifacts.

[**Real-Time Serving Pipeline:**](./serving/serving-graph.html) Rapid deployment of scalable data and ML pipelines using real-time serverless technology, including API handling, data preparation/enrichment, model serving, ensembles, driving and measuring actions, etc.

[**Real-Time monitoring:**](./model_monitoring/index.html) monitors data, models, resources, and production components and provides a feedback loop for exploring production data, identifying drift, alerting on anomalies or data quality issues, triggering retraining jobs, measuring business impact, etc.

`````


<a id="architecture"></a>
## MLRun Architecture

MLRun started as a community effort to map the different components in the ML project lifecycle, provide a common metadata layer, and automate the operationalization process (a.k.a MLOps).
 
Instead of a siloed, complex, and manual process, MLRun enables production pipeline design using a modular strategy, 
where the different parts contribute to a continuous, automated, and far simpler path from research and development to scalable 
production pipelines without refactoring code, adding glue logic, or spending significant efforts on data and ML engineering.

MLRun uses **Serverless Function** technology: write the code once, using your preferred development environment and 
simple "local" semantics, and then run it as-is on different platforms and at scale. MLRun automates the build process, execution, 
data movement, scaling, versioning, parameterization, output tracking, CI/CD integration, deployment to production, monitoring, and more. 

Those easily developed data or ML "functions" can then be published or loaded from a marketplace and used later to form offline or real-time 
production pipelines with minimal engineering efforts.

<p align="center"><img src="_static/images/mlrun-flow.png" alt="mlrun-flow" width="600"/></p><br>

### MLRun: An Integrated and Open Approach

Data preparation, model development, model and application delivery, and end to end monitoring are tightly connected: 
they cannot be managed in silos. This is where MLRun MLOps orchestration comes in. ML, data, and DevOps/MLOps teams 
collaborate using the same set of tools, practices, APIs, metadata, and version control.

MLRun provides an open architecture that supports your existing development tools, services, and practices through an open API/SDK and pluggable architecture. 

<b>MLRun simplifies & accelerates the time to production !</b>

<img src="_static/images/pipeline.png" alt="pipeline"/>

<br><br>

While each component in MLRun is independent, the integration provides much greater value and simplicity. For example:
- The training jobs obtain features from the feature store and update the feature store with metadata, which will be used in the serving or monitoring.
- The real-time pipeline enriches incoming events with features stored in the feature store. It can also use feature metadata (policies, statistics, schema, etc.) to impute missing data or validate data quality.
- The monitoring layer collects real-time inputs and outputs from the real-time pipeline and compares them with the features data/metadata from the feature store or model metadata generated by the training layer. Then, it writes all the fresh production data back to the feature store so it can be used for various tasks such as data analysis, model retraining (on fresh data), and model improvements.

When one of the components detailed above is updated, it immediately impacts the feature generation, the model serving pipeline, and the monitoring. MLRun applies versioning to each component, as well as versioning and rolling upgrades across components.

