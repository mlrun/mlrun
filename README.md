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

MLRun is an open source AI orchestration platform for quickly building and managing continuous (gen) AI applications across their lifecycle. MLRun integrates into your development and CI/CD environment and automates the delivery of production data, ML pipelines, and online applications. 
MLRun significantly reduces engineering efforts, time to production, and computation resources. 
With MLRun, you can choose any IDE on your local machine or on the cloud. MLRun breaks the silos between data, ML, software, and DevOps/MLOps teams, enabling collaboration and fast continuous improvements.

Get started with the MLRun [**Tutorials and Examples**](https://docs.mlrun.org/en/stable/tutorials/index.html) and the [**Installation and setup guide**](https://docs.mlrun.org/en/stable/install.html), or read about the [**MLRun Architecture**](https://docs.mlrun.org/en/stable/architecture.html).

This page explains how MLRun addresses the [**gen AI tasks**](#genai-tasks), [**MLOps tasks**](#mlops-tasks), and presents the [**MLRun core components**](#core-components).

See the supported data stores, development tools, services, platforms, etc., supported by MLRun's open architecture in **https://docs.mlrun.org/en/stable/ecosystem.html**.

## Gen AI tasks

<p align="center"><img src="https://github.com/mlrun/mlrun/raw/development/docs/_static/images/ai-tasks.png" alt="ai-tasks" width="800"/></p><br>

Use MLRun to develop, scale, deploy, and monitor your AI model across your enterprise. The [**gen AI development workflow**](https://docs.mlrun.org/en/stable/genai/genai-flow.html) 
section describes the different tasks and stages in detail.

### Data management


MLRun supports batch or realtime data processing at scale, data lineage and versioning, structured and unstructured data, and more. 
Removing inappropriate data at an early stage saves resources that would otherwise be required later on.


**Docs:**
[Using LLMs to process unstructured data](https://docs.mlrun.org/en/stable/genai/data-mgmt/unstructured-data.html)
[Vector databases](https://docs.mlrun.org/en/stable/genai/data-mgmt/vector-databases.html)
[Guardrails for data management](https://docs.mlrun.org/en/stable/genai/data-mgmt/guardrails-data.html)
**Demo:**
[Call center demo](https://github.com/mlrun/demo-call-center>`
**Video:**
[Call center](https://youtu.be/YycMbxRgLBA>`

### Development
Use MLRun to build an automated ML pipeline to: collect data, 
preprocess (prepare) the data, run the training pipeline, and evaluate the model.

**Docs:**
[Working with RAG](https://docs.mlrun.org/en/stable/genai/development/working-with-rag.html), [Evalating LLMs](https://docs.mlrun.org/en/stable/genai/development/evaluating-llms.html), [Fine tuning LLMS](https://docs.mlrun.org/en/stable/genai/development/fine-tuning-llms.html)
**Demos:**
[Call center demo](https://github.com/mlrun/demo-call-center), [Build & deploy custom (fine-tuned) LLM models and applications](https://github.com/mlrun/demo-llm-tuning/blob/main), [Interactive bot demo using LLMs](https://github.com/mlrun/demo-llm-bot/blob/main)
**Video:**
[Call center](https://youtu.be/YycMbxRgLBA)


### Deployment
MLRun serving can productize the newly trained LLM as a serverless function using real-time auto-scaling Nuclio serverless functions. 
The application pipeline includes all the steps from accepting events or data, contextualizing it with a state  preparing the required model features, 
inferring results using one or more models, and driving actions. 


**Docs:**
[Serving gen AI models](https://docs.mlrun.org/en/stable/genai/deployment/genai_serving.html), GPU utilization](https://docs.mlrun.org/en/stable/genai/deployment/gpu_utilization.html), [Gen AI realtime serving graph](https://docs.mlrun.org/en/stable/genai/deployment/genai_serving_graph.html)
**Tutorial:**
[Deploy LLM using MLRun](https://docs.mlrun.org/en/stable/tutorials/genai_01_basic_tutorial.html)
**Demos:**
[Call center demo](https://github.com/mlrun/demo-call-center), [Build & deploy custom(fine-tuned)]LLM models and applications <https://github.com/mlrun/demo-llm-tuning/blob/main), [Interactive bot demo using LLMs]<https://github.com/mlrun/demo-llm-bot/blob/main)
**Video:**
[Call center]<https://youtu.be/YycMbxRgLBA)


### Live Ops
Monitor all resources, data, model and application metrics to ensure performance. Then identify risks, control costs, and measure business KPIs.
Collect production data, metadata, and metrics to tune the model and application further, and to enable governance and explainability.


**Docs:**
[Model monitoring <monitoring](https://docs.mlrun.org/en/stable/concepts/monitoring.html), [Alerts and notifications](https://docs.mlrun.org/en/stable/concepts/alerts-notifications.html)
**Tutorials:**
[Deploy LLM using MLRun](https://docs.mlrun.org/en/stable/tutorials/genai_01_basic_tutorial.html), [Model monitoring using LLM](https://docs.mlrun.org/en/stable/tutorials/genai-02-monitoring-llm.html)
**Demo:**
[Build & deploy custom (fine-tuned) LLM models and applications](https://github.com/mlrun/demo-llm-tuning/blob/main)


<a id="mlops-tasks"></a>
## MLOps tasks

<p align="center"><img src="https://github.com/mlrun/mlrun/raw/development/docs/_static/images/mlops-task.png" alt="mlrun-tasks" width="800"/></p><br>

The [**MLOps development workflow**](https://docs.mlrun.org/en/stable/mlops-dev-flow.html) section describes the different tasks and stages in detail.
MLRun can be used to automate and orchestrate all the different tasks or just specific tasks (and integrate them with what you have already deployed).

### Project management and CI/CD automation

In MLRun the assets, metadata, and services (data, functions, jobs, artifacts, models, secrets, etc.) are organized into projects.
Projects can be imported/exported as a whole, mapped to git repositories or IDE projects (in PyCharm, VSCode, etc.), which enables versioning, collaboration, and CI/CD. 
Project access can be restricted to a set of users and roles. 

**Docs:** [Projects and Automation](https://docs.mlrun.org/en/stable/projects/project.html), [CI/CD Integration](https://docs.mlrun.org/en/stable/projects/ci-integration.html)
**Tutorials:** [Quick start](https://docs.mlrun.org/en/stable/tutorials/01-mlrun-basics.html), [Automated ML Pipeline](https://docs.mlrun.org/en/stable/tutorials/04-pipeline.html)
**Video:** [Quick start](https://youtu.be/xI8KVGLlj7Q).

### Ingest and process data

MLRun provides abstract interfaces to various offline and online [**data sources**](https://docs.mlrun.org/en/stable/store/datastore.html), supports batch or realtime data processing at scale, data lineage and versioning, structured and unstructured data, and more. 
In addition, the MLRun [**Feature Store**](https://docs.mlrun.org/en/stable/feature-store/feature-store.html) automates the collection, transformation, storage, catalog, serving, and monitoring of data features across the ML lifecycle and enables feature reuse and sharing. 

See: **Docs:** [Ingest and process data](https://docs.mlrun.org/en/stable/data-prep/index.html), [Feature Store](https://docs.mlrun.org/en/stable/feature-store/feature-store.html), [Data & Artifacts](https://docs.mlrun.org/en/stable/concepts/data.html)
**Tutorials:** [Quick start](https://docs.mlrun.org/en/stable/tutorials/01-mlrun-basics.html), [Feature Store](https://docs.mlrun.org/en/stable/feature-store/basic-demo.html).

### Develop and train models

MLRun allows you to easily build ML pipelines that take data from various sources or the Feature Store and process it, train models at scale with multiple parameters, test models, tracks each experiments, register, version and deploy models, etc. MLRun provides scalable built-in or custom model training services, integrate with any framework and can work with 3rd party training/auto-ML services. You can also bring your own pre-trained model and use it in the pipeline.

**Docs:** [Develop and train models](https://docs.mlrun.org/en/stable/development/index.html), [Model Training and Tracking](https://docs.mlrun.org/en/stable/development/model-training-tracking.html), [Batch Runs and Workflows](https://docs.mlrun.org/en/stable/concepts/runs-workflows.html)
**Tutorials:** [Train, compare, and register models](https://docs.mlrun.org/en/stable/tutorials/02-model-training.html), [Automated ML Pipeline](https://docs.mlrun.org/en/stable/tutorials/04-pipeline.html)
**Video:** [Train and compare models](https://youtu.be/bZgBsmLMdQo).

### Deploy models and applications

MLRun rapidly deploys and manages production-grade real-time or batch application pipelines using elastic and resilient serverless functions. MLRun addresses the entire ML application: intercepting application/user requests, running data processing tasks, inferencing using one or more models, driving actions, and integrating with the application logic.

**Docs:** [Deploy models and applications](https://docs.mlrun.org/en/stable/deployment/index.html), [Realtime Pipelines](https://docs.mlrun.org/en/stable/serving/serving-graph.html), [Batch Inference](https://docs.mlrun.org/en/stable/deployment/batch_inference.html)
**Tutorials:** [Realtime Serving](https://docs.mlrun.org/en/stable/tutorials/03-model-serving.html), [Batch Inference](https://docs.mlrun.org/en/stable/tutorials/07-batch-infer.html), [Advanced Pipeline](https://docs.mlrun.org/en/stable/tutorials/07-batch-infer.html)
**Video:** [Serving pre-trained models](https://youtu.be/OUjOus4dZfw).

### Model Monitoring

Observability is built into the different MLRun objects (data, functions, jobs, models, pipelines, etc.), eliminating the need for complex integrations and code instrumentation. With MLRun, you can observe the application/model resource usage and model behavior (drift, performance, etc.), define custom app metrics, and trigger alerts or retraining jobs.

**Docs:** [Model monitoring](https://docs.mlrun.org/en/stable/concepts/model-monitoring.html), [Model Monitoring Overview](https://docs.mlrun.org/en/stable/monitoring/model-monitoring-deployment.html)
**Tutorials:** [Model Monitoring & Drift Detection](https://docs.mlrun.org/en/stable/tutorials/05-model-monitoring.html).


<a id="core-components"></a>
## MLRun core components

<p align="center"><img src="https://github.com/mlrun/mlrun/raw/development/docs/_static/images/mlops-core.png" alt="mlrun-core" width="800"/></p><br>


MLRun includes the following major components:

[**Project Management:**](https://docs.mlrun.org/en/stable/projects/project.html) A service (API, SDK, DB, UI) that manages the different project assets (data, functions, jobs, workflows, secrets, etc.) and provides central control and metadata layer.  

[**Functions:**](https://docs.mlrun.org/en/stable/runtimes/functions.html) automatically deployed software package with one or more methods and runtime-specific attributes (such as image, libraries, command, arguments, resources, etc.).

[**Data & Artifacts:**](https://docs.mlrun.org/en/stable/concepts/data.html) Glueless connectivity to various data sources, metadata management, catalog, and versioning for structures/unstructured artifacts.

[**Batch Runs & Workflows:**](https://docs.mlrun.org/en/stable/concepts/runs-workflows.html) Execute one or more functions with specific parameters and collect, track, and compare all their results and artifacts.

[**Real-Time Serving Pipeline:**](https://docs.mlrun.org/en/stable/serving/serving-graph.html) Rapid deployment of scalable data and ML pipelines using real-time serverless technology, including API handling, data preparation/enrichment, model serving, ensembles, driving and measuring actions, etc.

[**Model monitoring:**](https://docs.mlrun.org/en/stable/monitoring/index.html) monitors data, models, resources, and production components and provides a feedback loop for exploring production data, identifying drift, alerting on anomalies or data quality issues, triggering retraining jobs, measuring business impact, etc.

[**Alerts and notifications:**](https://docs.mlrun.org/en/stable/concepts/model-monitoring.html) Use alerts to identify and inform you of possible problem situations. Use notifications to report status on runs and pipelines.

[**Feature Store:**](https://docs.mlrun.org/en/stable/feature-store/feature-store.html) automatically collects, prepares, catalogs, and serves production data features for development (offline) and real-time (online) deployment using minimal engineering effort.