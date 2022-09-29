(architecture)=
# Using MLRun 

```{div} full-width
MLRun is an open MLOps platform for quickly building and managing continuous ML applications across their lifecycle. MLRun integrates into your development and CI/CD environment and automates the delivery of production data, ML pipelines, and online applications, significantly reducing engineering efforts, time to production, and computation resources. 
MLRun breaks the silos between data, ML, software, and DevOps/MLOps teams, enabling collaboration and fast continuous improvements.
```

- [**MLOps Tasks**](#mlops-tasks)
- [**MLRun Core components and Architecture**](#core-components)
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

```{grid-item-card} Continious Development and Training 
:text-align: center
:link: ./model_development/continuous-model-development.html
```

```{grid-item-card} Model and ML Application Deployment
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

{octicon}`code-square`
Tutorials:
{bdg-link-primary}`quick start <./tutorial/01-mlrun-basics.html>`
{bdg-link-primary}`Automated ML Pipeline <./tutorial/04-pipeline.html>`


### Data Ingestion and Processing

MLRun provides abstract interfaces to various offline and online [**data sources**](./concepts/data-feature-store.html), supports batch or real-time data processing at scale, data lineage and versioning, and more. 
In addition, MLRun [**Feature Store**](./feature-store/feature-store.html) automates the collection, transformation, storage, catalog, serving, and monitoring of data features across the ML lifecycle and enables feature reuse and sharing.
{bdg-link-primary-line}`more... <./data-collect-prep/ingest-process-data.html>`

{octicon}`mortar-board`
Docs:
{bdg-link-info}`Feature Store <./feature-store/feature-store.html>`
{bdg-link-info}`Data & Artifacts <./concepts/data-feature-store.html>`
, {octicon}`code-square`
Tutorials:
{bdg-link-primary}`quick start <./tutorial/01-mlrun-basics.html>`
{bdg-link-primary}`Feature Store <./feature-store/basic-demo.html>`
, {octicon}`video`
Videos:
{bdg-link-warning}`getting started <./tutorial/01-mlrun-basics.html>`

### Continious Development and Training

MLRun allows you to build ML pipelines that take data from various sources or the Feature Store, process and validate the data, train the model at scale with multiple parameters, test the model, register in the model repository, etc. MLRun provides scalable built-in or custom model training services or can work with 3rd party training/auto-ML services.

MLRun orchestrates the execution of the pipelines, tracks the experiments and results, versions the data, and integrates natively with CI/CD frameworks.
{bdg-link-primary-line}`more... <./model_development/continuous-model-development.html>`

{octicon}`code-square`
Tutorials:
{bdg-link-primary}`Train, Compare, and Register Models <./tutorial/02-model-training.html>`
{bdg-link-primary}`Automated ML Pipeline <./tutorial/04-pipeline.html>`

### Deploy models & ML applications

It is possible to run models using online serving as well as batch inference. MLRun takes model execution a step further, and allow you to define a whole pipeline along with the model. This allows you to perform additional tasks, such as data manipulation prior to running a model as well as act upon the result of the model (e.g., for decisioning or for using the model output as a feature of other models).

### Model & data monitoring

Monitoring is built-in and is easy to set-up. One of the key advantages of using MLRun is that you don't have to create your own monitoring solution, as you can easily get operational, model and data results and alerts with an easy-to-use API. As an added bonus, you get monitoring UI out-of-the-box.

`````


## MLRun Core components and Architecture


`````{div} full-width

````{grid} 6
:gutter: 2

```{grid-item-card} Project Management & Automation
:columns: 12
:text-align: center
:link: ./projects/project.html
```

```{grid-item-card} Serverless Functions
:columns: 6 4 4 2
:text-align: center
:link: ./runtimes/functions.html
```

```{grid-item-card} Batch Runs & Workflows 
:columns: 6 4 4 2
:text-align: center
:link: ./projects/workflows.html
```

```{grid-item-card} Real-time Pipelines
:columns: 6 4 4 2
:text-align: center
:link: ./serving/serving-graph.html
```

```{grid-item-card} Data & Artifacts
:columns: 6 4 4 2
:text-align: center
:link: concepts/data-feature-store.html
```

```{grid-item-card} Feature Store
:columns: 6 4 4 2
:text-align: center
:link: ./feature-store/feature-store.html
```

```{grid-item-card} Monitoring
:columns: 6 4 4 2
:text-align: center
:link: ./model_monitoring/index.html
```

````
`````








<a id="the-challenge"></a>
### The challenge

Most data science solutions and platforms on the market today begin and therefore emphasize the research workflow. 
When it comes time to integrate the generated models into real-world AI applications, they have significant functionality gaps.

These types of solutions tend to be useful only for the model development flow, and contribute very little to the production pipeline: 
automated data collection and preparation, automated training and evaluation pipelines, real-time application pipelines, 
data quality and model monitoring, feedback loops, etc.

To address the full ML application lifecycle, it’s common for organizations to combine many different tools which then forces 
them to develop and maintain complex glue layers, introduce manual processes, and creates technology silos that slow down 
developers and data scientists. 

With this disjointed approach, the ML team must re-engineer the entire flow to fit production environments and methodologies 
while consuming excessive resources. Organizations need a way to streamline the process, 
automate as many tasks as possible, and break the silos between data, ML, and DevOps/MLOps teams.

<a id="why-mlrun"></a>
### MLRun - The Open Source MLOps Orchestration

Instead of this siloed, complex and manual process, MLRun enables production pipeline design using a modular strategy, 
where the different parts contribute to a continuous, automated, and far simpler path from research and development to scalable 
production pipelines, without refactoring code, adding glue logic, or spending significant efforts on data and ML engineering.

MLRun uses **Serverless Function** technology: write the code once, using your preferred development environment and 
simple "local" semantics, and then run it as-is on different platforms and at scale. MLRun automates the build process, execution, 
data movement, scaling, versioning, parameterization, outputs tracking, CI/CD integration, deployment to production, monitoring, and more. 

Those easily developed data or ML "functions" can then be published or loaded from a marketplace and used later to form offline or real-time 
production pipelines with minimal engineering efforts.

<p align="center"><img src="_static/images/mlrun-flow.png" alt="mlrun-flow" width="600"/></p><br>

Data preparation, model development, model and application delivery, and end to end monitoring are tightly connected: 
they cannot be managed in silos. This is where MLRun MLOps orchestration comes in. ML, data, and DevOps/MLOps teams 
collaborate using the same set of tools, practices, APIs, metadata, and version control.

<b>MLRun simplifies & accelerates the time to production.</b>

### Architecture 

<img src="_static/images/pipeline.png" alt="pipeline"/>

<br><br>
MLRun is composed of the following layers:

- **{ref}`Feature Store <feature-store>`** &mdash; collects, prepares, catalogs, and serves data features for development (offline) and real-time (online) 
usage for real-time and batch data.
- **{ref}`ML CI/CD pipeline <ci-integration>`** &mdash; automatically trains, tests, optimizes, and deploys or updates models using a snapshot of the production 
data (generated by the feature store) and code from the source control (Git).
- **{ref}`Real-Time Serving Pipeline <serving>`** &mdash; Rapid deployment of scalable data and ML pipelines using real-time serverless technology, including 
the API handling, data preparation/enrichment, model serving, ensembles, driving and measuring actions, etc.
- **{ref}`Real-Time monitoring and retraining <model_monitoring>`** &mdash; monitors data, models, and production components and provides a feedback loop for exploring production data, identifying drift, alerting on anomalies or data quality issues, triggering re-training jobs, measuring business impact, etc.

While each of those layers is independent, the integration provides much greater value and simplicity. For example:
- The training jobs obtain features from the feature store and update the feature store with metadata, which will be used in the serving or monitoring.
- The real-time pipeline enriches incoming events with features stored in the feature store, and can also use feature metadata (policies, statistics, schema, etc.) to impute missing data or validate data quality.
- The monitoring layer collects real-time inputs and outputs from the real-time pipeline and compares them with the features data/metadata from the feature store or model metadata generated by the training layer. It writes all the fresh production data back to the feature store so it can be used for various tasks such as data analysis, model re-training (on fresh data), model improvements.

When one of the components detailed above is updated, it immediately impacts the feature generation, the model serving pipeline, and the monitoring. MLRun applies versioning to each component, as well as versioning and rolling upgrades across components.

<a id="basic-components"></a>
### Basic components

MLRun has the following main components that are used throughout the system:

- <a id="def-project"></a>**Project** &mdash; a container for organizing all of your work on a particular activity.
    Projects consist of metadata, source code, workflows, data and artifacts, models, triggers, and member management for user collaboration. Read more in [Projects](./projects/project.html).

- <a id="def-function"></a>**Function** &mdash; a software package with one or more methods and runtime-specific attributes (such as image, command, arguments, and environment). Read more in [MLRun serverless functions](./concepts/functions-concepts.html) and {ref}`functions`.

- <a id="def-run"></a>**Run** &mdash; an object that contains information about an executed function.
    The run object is created as a result of running a function, and contains the function attributes (such as arguments, inputs, and outputs), as well the execution status and results (including links to output artifacts). Read more in {ref}`submitting-tasks-jobs-to-functions`.

- <a id="def-artifact"></a>**Artifact** &mdash; versioned data artifacts (such as data sets, files and models) are produced or consumed by functions, runs, and workflows. Read more in [Artifacts](./store/artifacts.html).

- <a id="def-workflow"></a>**Workflow** &mdash; defines a functions pipeline or a directed acyclic graph (DAG) to execute using [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-quickstart/)
  or MLRun [Real-time serving pipelines](./serving/serving-graph.html). Read more in [Workflows](./projects/workflows.html).
  
- **UI** &mdash; a graphical user interface (dashboard) for displaying and managing projects and their contained experiments, artifacts, and code.