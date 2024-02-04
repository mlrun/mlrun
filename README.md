<a id="top"></a>
[![Build Status](https://github.com/mlrun/mlrun/actions/workflows/build.yaml/badge.svg?branch=development)](https://github.com/mlrun/mlrun/actions/workflows/build.yaml?query=branch%3Adevelopment)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version fury.io](https://badge.fury.io/py/mlrun.svg)](https://pypi.python.org/pypi/mlrun/)
[![Documentation](https://readthedocs.org/projects/mlrun/badge/?version=latest)](https://mlrun.readthedocs.io/en/latest/?badge=latest)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/mlrun/mlrun)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/mlrun/mlrun?sort=semver)
[![Join MLOps Live](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://mlopslive.slack.com)

<p align="left"><img src="https://github.com/mlrun/mlrun/raw/development/docs/_static/images/MLRun-logo.png" alt="MLRun logo" width="150"/></p>

# Using MLRun 

MLRun is an open MLOps platform for quickly building and managing continuous ML applications across their lifecycle. MLRun integrates into your development and CI/CD environment and automates the delivery of production data, ML pipelines, and online applications, significantly reducing engineering efforts, time to production, and computation resources.
With MLRun, you can choose any IDE on your local machine or on the cloud. MLRun breaks the silos between data, ML, software, and DevOps/MLOps teams, enabling collaboration and fast continuous improvements.

Get started with MLRun [**Tutorials and Examples**](https://docs.mlrun.org/en/latest/tutorials/index.html), [**Installation and setup guide**](https://docs.mlrun.org/en/latest/install.html), or read about [**MLRun Architecture**](https://docs.mlrun.org/en/latest/architecture.html).

This page explains how MLRun addresses the [**MLOps Tasks**](#mlops-tasks) and the [**MLRun core components**](#core-components).

<a id="mlops-tasks"></a>
## MLOps tasks

<p align="center"><img src="https://github.com/mlrun/mlrun/raw/development/docs/_static/images/mlops-task.png" alt="mlrun-tasks" width="800"/></p><br>

The [**MLOps development workflow**](https://docs.mlrun.org/en/latest/mlops-dev-flow.html) section describes the different tasks and stages in detail.
MLRun can be used to automate and orchestrate all the different tasks or just specific tasks (and integrate them with what you have already deployed).

### Project management and CI/CD automation

In MLRun the assets, metadata, and services (data, functions, jobs, artifacts, models, secrets, etc.) are organized into projects.
Projects can be imported/exported as a whole, mapped to git repositories or IDE projects (in PyCharm, VSCode, etc.), which enables versioning, collaboration, and CI/CD. 
Project access can be restricted to a set of users and roles. 

See: **Docs:** [Projects and Automation](https://docs.mlrun.org/en/latest/projects/project.html), [CI/CD Integration](https://docs.mlrun.org/en/latest/projects/ci-integration.html), **Tutorials:** [Quick start](https://docs.mlrun.org/en/latest/tutorials/01-mlrun-basics.html), [Automated ML Pipeline](https://docs.mlrun.org/en/latest/tutorials/04-pipeline.html), **Video:** [quick start](https://youtu.be/xI8KVGLlj7Q).

### Ingest and process data

MLRun provides abstract interfaces to various offline and online [**data sources**](https://docs.mlrun.org/en/latest/store/datastore.html), supports batch or realtime data processing at scale, data lineage and versioning, structured and unstructured data, and more. 
In addition, the MLRun [**Feature Store**](https://docs.mlrun.org/en/latest/feature-store/feature-store.html) automates the collection, transformation, storage, catalog, serving, and monitoring of data features across the ML lifecycle and enables feature reuse and sharing. 

See: **Docs:** [Ingest and process data](https://docs.mlrun.org/en/latest/data-prep/index.html), [Feature Store](https://docs.mlrun.org/en/latest/feature-store/feature-store.html), [Data & Artifacts](https://docs.mlrun.org/en/latest/concepts/data.html); **Tutorials:** [Quick start](https://docs.mlrun.org/en/latest/tutorials/01-mlrun-basics.html), [Feature Store](https://docs.mlrun.org/en/latest/feature-store/basic-demo.html).

### Develop and train models

MLRun allows you to easily build ML pipelines that take data from various sources or the Feature Store and process it, train models at scale with multiple parameters, test models, tracks each experiments, register, version and deploy models, etc. MLRun provides scalable built-in or custom model training services, integrate with any framework and can work with 3rd party training/auto-ML services. You can also bring your own pre-trained model and use it in the pipeline.

See: **Docs:** [Develop and train models](https://docs.mlrun.org/en/latest/development/index.html), [Model Training and Tracking](https://docs.mlrun.org/en/latest/development/model-training-tracking.html), [Batch Runs and Workflows](https://docs.mlrun.org/en/latest/concepts/runs-workflows.html); **Tutorials:** [Train & Eval Models](https://docs.mlrun.org/en/latest/tutorials/02-model-training.html), [Automated ML Pipeline](https://docs.mlrun.org/en/latest/tutorials/04-pipeline.html); **Video:** [Training models](https://youtu.be/bZgBsmLMdQo).

### Deploy models and applications

MLRun rapidly deploys and manages production-grade real-time or batch application pipelines using elastic and resilient serverless functions. MLRun addresses the entire ML application: intercepting application/user requests, running data processing tasks, inferencing using one or more models, driving actions, and integrating with the application logic.

See: **Docs:** [Deploy models and applications](https://docs.mlrun.org/en/latest/deployment/index.html), [Realtime Pipelines](https://docs.mlrun.org/en/latest/serving/serving-graph.html), [Batch Inference](https://docs.mlrun.org/en/latest/concepts/TBD.html), **Tutorials:** [Realtime Serving](https://docs.mlrun.org/en/latest/tutorials/03-model-serving.html), [Batch Inference](https://docs.mlrun.org/en/latest/tutorials/07-batch-infer.html), [Advanced Pipeline](https://docs.mlrun.org/en/latest/tutorials/07-batch-infer.html); **Video:** [Serving models](https://youtu.be/OUjOus4dZfw).

### Monitor and alert

Observability is built into the different MLRun objects (data, functions, jobs, models, pipelines, etc.), eliminating the need for complex integrations and code instrumentation. With MLRun, you can observe the application/model resource usage and model behavior (drift, performance, etc.), define custom app metrics, and trigger alerts or retraining jobs.

See: **Docs:** [Monitor and alert](https://docs.mlrun.org/en/latest/monitoring/index.html), [Model Monitoring Overview](https://docs.mlrun.org/en/latest/monitoring/model-monitoring-deployment.html), **Tutorials:** [Model Monitoring & Drift Detection](https://docs.mlrun.org/en/latest/tutorials/05-model-monitoring.html).


<a id="core-components"></a>
## MLRun core components

<p align="center"><img src="https://github.com/mlrun/mlrun/raw/development/docs/_static/images/mlops-core.png" alt="mlrun-core" width="800"/></p><br>

MLRun includes the following major components:

[**Project Management:**](https://docs.mlrun.org/en/latest/projects/project.html) A service (API, SDK, DB, UI) that manages the different project assets (data, functions, jobs, workflows, secrets, etc.) and provides central control and metadata layer.  

[**Serverless Functions:**](https://docs.mlrun.org/en/latest/runtimes/functions.html) automatically deployed software package with one or more methods and runtime-specific attributes (such as image, libraries, command, arguments, resources, etc.).

[**Data & Artifacts:**](https://docs.mlrun.org/en/latest/concepts/data-feature-store.html) Glueless connectivity to various data sources, metadata management, catalog, and versioning for structures/unstructured artifacts.

[**Feature Store:**](https://docs.mlrun.org/en/latest/feature-store/feature-store.html) automatically collects, prepares, catalogs, and serves production data features for development (offline) and real-time (online) deployment using minimal engineering effort.

[**Batch Runs & Workflows:**](https://docs.mlrun.org/en/latest/concepts/runs-workflows.html) Execute one or more functions with specific parameters and collect, track, and compare all their results and artifacts.

[**Real-Time Serving Pipeline:**](https://docs.mlrun.org/en/latest/serving/serving-graph.html) Rapid deployment of scalable data and ML pipelines using real-time serverless technology, including API handling, data preparation/enrichment, model serving, ensembles, driving and measuring actions, etc.

[**Real-Time monitoring:**](https://docs.mlrun.org/en/latest/monitoring/index.html) monitors data, models, resources, and production components and provides a feedback loop for exploring production data, identifying drift, alerting on anomalies or data quality issues, triggering retraining jobs, measuring business impact, etc.
