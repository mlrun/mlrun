.. mlrun documentation master file, created by
   sphinx-quickstart on Thu Jan  2 15:59:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLRun Package Documentation
============================

Introduction
============

MLRun is a generic and convenient mechanism for data scientists and software developers to describe and run tasks related to machine learning (ML) in various, scalable runtime environments and ML pipelines while automatically tracking executed code, metadata, inputs, and outputs.
MLRun integrates with the `Nuclio <https://nuclio.io/>`_ serverless project and with `Kubeflow Pipelines <https://github.com/kubeflow/pipelines>`_.

MLRun features a Python package (``mlrun``), a command-line interface (``mlrun``), and a graphical user interface (the MLRun dashboard).

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quick-start
   general
   install
   external/remote.md
   end-to-end-pipeline
   data-management-and-versioning
   projects
   load-from-marketplace
   job-submission-and-tracking
   model-management-and-serving
   examples
   api/index

