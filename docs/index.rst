.. mlrun documentation master file, created by
   sphinx-quickstart on Thu Jan  2 15:59:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLRun Package Documentation
============================

*The Open-Source MLOps Orchestration Framework.*

Introduction
************

**MLRun** offers an integrative approach to manage your machine-learning pipelines from early development through management in your production environment. MLRun offers a convenient abstraction layer to a wide variety of technology stacks while empowering the Data Engineers and Data Scientists to define the feature and models.

Key Benefits
------------

MLRun provides the following key benefits:

- **Rapid deployment** of code to production pipelines
- **Elastic scaling** for batch and real-time workloads
- **Feature management** ingestion, preparation, and monitoring
- **Works anywhere** in your local IDE, multi-cloud or on-prem

Key Features
--------------

MLRun includes the following key features:

- **Feature store** – Define and reuse features with a robust feature store that includes a highly flexible transformation framework.
- **Elastic serverless runtimes** – Turn your Python code to composable functions, that can run at scale on Kubernetes, Dask and Horovod in a single command.
- **Function marketplace** – Leverage a function marketplace to accelerate your model development process.
- **Data ingestion & preparation** – Read and transform data from batch and online data stores.
- **Model training & testing** – Train models at scale with automated testing functions.
- **Real-time Data & model Pipeline** – Deploy real-time pipelines for data collection, model serving and monitoring.
- **Data & model Monitoring** – Automate model monitoring and drift detection.
- **Central data & metadata management** – Log all data, models and artifacts and track all code execution.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick-start
   general
   install
   remote
   end-to-end-pipeline
   data-management-and-versioning
   projects
   load-from-marketplace
   job-submission-and-tracking
   serving/index
   examples
   cli
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
