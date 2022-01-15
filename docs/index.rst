.. mlrun documentation master file, created by
   sphinx-quickstart on Thu Jan  2 15:59:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLRun Package Documentation
============================

*The Open-Source MLOps Orchestration Framework*

Introduction
************

MLRun is an open-source MLOps framework that offers an integrative approach to managing your machine-learning pipelines from early development through model development to full pipeline deployment in production.
MLRun offers a convenient abstraction layer to a wide variety of technology stacks while empowering data engineers and data scientists to define the feature and models.

The MLRun Architecture
----------------------

.. image:: _static/images/mlrun-architecture.png
    :alt: MLRun architecture

MLRun is composed of the following layers:

- **Feature and Artifact Store** – 
    handles the ingestion, processing, metadata, and storage of data and features across multiple repositories and technologies.
- **Elastic Serverless Runtimes** –
    converts simple code to scalable and managed microservices with workload-specific runtime engines (such as Kubernetes jobs, Nuclio, Dask, Spark, and Horovod).
- **ML Pipeline Automation** –
    automates data preparation, model training and testing, deployment of real-time production pipelines, and end-to-end monitoring.
- **Central Management** –
    provides a unified portal for managing the entire MLOps workflow.
    The portal includes a UI, a CLI, and an SDK, which are accessible from anywhere.

Review the relevant documentation sections to learn about each component.

Key Benefits
------------

MLRun provides the following key benefits:

- **Rapid deployment** of code to production pipelines
- **Elastic scaling** of batch and real-time workloads
- **Feature management** – ingestion, preparation, and monitoring
- **Works anywhere** – your local IDE, multi-cloud, or on-prem

Table Of Contents
-------------------

.. toctree::
   :maxdepth: 1
   :caption: MLRun Basics:

   quick-start
   tutorial/index
   architecture
   install
   howto/index

.. toctree::
   :maxdepth: 1
   :caption: Functions and ML Pipelines:

   runtimes/functions
   runtimes/distributed
   projects/overview
   secrets

.. toctree::
   :maxdepth: 1
   :caption: Online Pipelines & Serving:

   serving/index
   model_monitoring/index

.. toctree::
   :maxdepth: 1
   :caption: Feature Store:

   feature-store/feature-store
   feature-store/feature-sets
   feature-store/transformations
   feature-store/feature-vectors
   feature-store/training-serving
   feature-store/basic-demo
   feature-store/end-to-end-demo/index

.. toctree::
   :maxdepth: 1
   :caption: Artifact Management:

   store/datastore
   store/artifacts
   store/models

.. toctree::
   :maxdepth: 1
   :caption: References:

   examples
   cli
   genindex
   api/index
