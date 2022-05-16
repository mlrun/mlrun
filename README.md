<a id="top"></a>
[![Build Status](https://github.com/mlrun/mlrun/workflows/CI/badge.svg)](https://github.com/mlrun/mlrun/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version fury.io](https://badge.fury.io/py/mlrun.svg)](https://pypi.python.org/pypi/mlrun/)
[![Documentation](https://readthedocs.org/projects/mlrun/badge/?version=latest)](https://mlrun.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="left"><img src="docs/_static/images/MLRun-logo.png" alt="MLRun logo" width="150"/></p>



## MLRun - The Open Source MLOps Orchestration Framework

MLRun enables production pipeline design using a modular strategy, where the different parts contribute to a continuous, automated, and far simpler path from research and development to scalable production pipelines, without refactoring code, adding glue logic, or spending significant efforts on data and ML engineering.

MLRun uses **Serverless Function** technology: write the code once, using your preferred development environment and simple “local” semantics, and then run it as-is on different platforms and at scale. MLRun automates the build process, execution, data movement, scaling, versioning, parameterization, outputs tracking, CI/CD integration, deployment to production, monitoring, and more.

Those easily developed data or ML “functions” can then be published or loaded from a marketplace and used later to form offline or real-time production pipelines with minimal engineering efforts.

<p align="center"><img src="./docs/_static/images/mlrun-flow.png" alt="mlrun-flow" width="600"/></p><br>

Data preparation, model development, model and application delivery, and end to end monitoring are tightly connected: they cannot be managed in silos. This is where MLRun MLOps orchestration comes in. ML, data, and DevOps/MLOps teams collaborate using the same set of tools, practices, APIs, metadata, and version control.

**MLRun simplifies & accelerates the time to production.**

## Architecture 

![pipeline](./docs/_static/images/pipeline.png)

MLRun is composed of the following layers:

- **[Feature Store](./docs/feature-store/feature-store.html)** &mdash; collects, prepares, catalogs, and serves data features for development (offline) and real-time (online) usage for real-time and batch data. See also 
[Feature store: data ingestion](./docs/feature-store/feature-store-data-ingestion) and [Feature store: data retrieval](./docs/feature-store/feature-store-data-retrieval), as well as the [Feature Store tutorials](./docs/feature-store/feature-store-tutorials).
- **[ML CI/CD pipeline](.docs//projects/ci-integration)** &mdash; automatically trains, tests, optimizes, and deploys or updates models using a snapshot of the production 
data (generated by the feature store) and code from the source control (Git).
- **[Real-Time Serving Pipeline](./docs/serving/serving-graph)** &mdash; Rapid deployment of scalable data and ML pipelines using real-time serverless technology, including 
the API handling, data preparation/enrichment, [model serving](https://docs.mlrun.org/en/latest/serving/build-graph-model-serving.html), ensembles, driving and measuring actions, etc.
- **[Real-Time monitoring and retraining](.docs//model_monitoring/index)** &mdash; monitors data, models, and production components and provides a feedback loop for exploring production data, identifying drift, alerting on anomalies or data quality issues, triggering re-training jobs, measuring business impact, etc.

## Get started

It's easy to start using MLRun: 
1. Install MLRun using one of [over Kubernetes Cluster](https://docs.mlrun.org/en/latest/install/kubernetes.html) or [locally using Docker](https://docs.mlrun.org/en/latest/install/local-docker.html).<br>
   Alternatively, you can [Use the managed MLRun service](https://www.iguazio.com/docs/latest-release/).
2. [Set up your client environment](https://docs.mlrun.org/en/latest/install/remote.html) to work with the service. 
3. Use the [Quick Start tutorial](https://docs.mlrun.org/en/latest/quick-start/quick-start.html) to develop and deploy machine learning applications to production.<br>
For hands-on learning, try the [MLRun Katakoda Scenarios](https://www.katacoda.com/mlrun). And you can watch the [Tutorial on Youtube](https://www.youtube.com/embed/O6g1pJJ609U) to see the flow in action.

## MLRun documentation

Read more in the MLRun documentation, including:
- MLRun basics
   - [What is MLRun?](https://docs.mlrun.org/en/latest/index.html)
   - [Quick-Start Guide](https://docs.mlrun.org/en/latest/quick-start/quick-start.html)
   - [Tutorials and examples](https://docs.mlrun.org/en/latest/howto/index.html)
   - [Installation and setup guide](https://docs.mlrun.org/en/latest/install.html)
- Concepts
   - [Projects](https://docs.mlrun.org/en/latest/projects/project.html)
   - [MLRun serverless functions](https://docs.mlrun.org/en/latest/concepts/functions-concepts.html)
   - [Data stores and data items](https://docs.mlrun.org/en/latest/concepts/data-feature-store.html)
   - [Feature store](https://docs.mlrun.org/en/latest/feature-store/feature-store.html)
   - [Runs, functions, and workflows](https://docs.mlrun.org/en/latest/concepts/runs-experiments-workflows.html)
   - [Artifacts and models](https://docs.mlrun.org/en/latest/store/artifacts.html)
   - [Deployment and monitoring](https://docs.mlrun.org/en/latest/concepts/deployment-monitoring.html)
- Working with data
   - [Data ingestion](https://docs.mlrun.org/en/latest/feature-store/feature-store-data-ingestion.html)
   - [Data retrieval](https://docs.mlrun.org/en/latest/feature-store/feature-store-data-retrieval.html)
   - [Tutorials](https://docs.mlrun.org/en/latest/feature-store/feature-store-tutorials.html)
- Develop Functions and models
   - [Creating and using functions](https://docs.mlrun.org/en/latest/runtimes/functions.html)
   - [Run, track, and compare jobs](https://docs.mlrun.org/en/latest/runtimes/run-track-compare-jobs.html)
- Deploy ML applications
   - [Real-time serving pipelines (graphs)](https://docs.mlrun.org/en/latest/serving/serving-graph.html)
   - [Model serving pipelines](https://docs.mlrun.org/en/latest/serving/build-graph-model-serving.html)
   - [CI/CD, rolling upgrades, git](https://docs.mlrun.org/en/latest/model_monitoring/ci-cd-rolling-upgrades-git.html)
- References
   - [API](https://docs.mlrun.org/en/latest/api/index.html)
   - [CLI](https://docs.mlrun.org/en/latest/cli.html)
   - [Glossary](https://docs.mlrun.org/en/latest/glossary.html)
