.. mlrun documentation master file, created by
   sphinx-quickstart on Thu Jan  2 15:59:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLRun Package Documentation
============================

*The Open-Source MLOps Orchestration Framework*

MLRun is an open-source MLOps framework that offers an integrative approach to managing your machine-learning pipelines from early development through model development to full pipeline deployment in production.
MLRun offers a convenient abstraction layer to a wide variety of technology stacks while empowering data engineers and data scientists to define the feature and models.

The MLRun Architecture
************************

.. image:: _static/images/mlrun-architecture.png
    :alt: MLRun architecture

MLRun is composed of the following layers:

- **Feature and Artifact Store**  
    Handles the ingestion, processing, metadata, and storage of data and features across multiple repositories and technologies.
- **Elastic Serverless Runtimes**
    Converts simple code to scalable and managed microservices with workload-specific runtime engines (such as Kubernetes jobs, Nuclio, Dask, Spark, and Horovod).
- **ML Pipeline Automation**
    Automates data preparation, model training and testing, deployment of real-time production pipelines, and end-to-end monitoring.
- **Central Management**
    Provides a unified portal for managing the entire MLOps workflow.
    The portal includes a UI, a CLI, and an SDK, which are accessible from anywhere.

Key Benefits
************************

MLRun provides the following key benefits:

- **Rapid deployment** of code to production pipelines
- **Elastic scaling** of batch and real-time workloads
- **Feature management** – ingestion, preparation, and monitoring
- **Works anywhere** – your local IDE, multi-cloud, or on-prem

Table Of Contents
-------------------

.. toctree::
   :maxdepth: 1
   :caption: MLRun basics

   architecture
   mlops-dev-flow
   quick-start
   tutorial/index
      Add MLRun to existing code
      Work from IDE
      Additional demos / katacoda
   howto/index
   install
   
.. toctree::
   :maxdepth: 1
   :caption: Concepts
   
   projects/project
   concepts/functions-concepts
   concepts/data-feature-store
   concepts/runs-experiments-workflows
   store/artifacts
   concepts/deployment-monitoring

.. toctree::
   :maxdepth: 1
   :caption: Working with data

   <!--- feature-store/data-access-versioning --->
   <!--- feature-store/prepare-analyze-data --->
   feature-store/feature-store-data-ingestion
   feature-store/feature-store-data-retrieval
   feature-store/feature-store-tutorials
  
.. toctree::
   :maxdepth: 1
   :caption: Develop functions and models
   
   runtimes/functions
   runtimes/run-track-compare-jobs
   <!-- runtimes/develop-ml-models -->
   <!-- runtimes/develop-dl-nlp-models -->
   <!-- runtimes/run-multistage-workflows -->
   <!-- runtimes/manage-monitor-resources -->
      
.. toctree::
   :maxdepth: 1
   :caption: Deploy ML applications
   
   <!-- model_monitoring/model-registry-mgmt -->
   serving/serving-graph
   serving/build-graph-model-serving
   model_monitoring/index
   <!-- model_monitoring/ci-cd-rolling-upgrades-git -->

.. toctree::
   :maxdepth: 1
   :caption: References

   genindex
   api/index
   <!-- web-apps -->
   cli
   examples
   Glossary