(tutorial)=
# Getting Started Tutorial

This tutorial provides a hands-on introduction to using MLRun to implement a data science workflow and automate machine-learning operations (MLOps).

The tutorial covers MLRun fundamentals such as creation of projects and data ingestion and preparation, and demonstrates how to create an end-to-end machine-learning (ML) pipeline.
MLRun is integrated as a default (pre-deployed) shared service in the Iguazio Data Science Platform.

You'll learn how to

- Collect (ingest), prepare, and analyze data
- Train, deploy, and monitor an ML model
- Create and run an automated ML pipeline

You'll also learn about the basic concepts, components, and APIs that allow you to perform these tasks, including

- Setting up MLRun
- Creating and working with projects
- Creating, deploying and running MLRun functions
- Using MLRun to run functions, jobs, and full workflows
- Deploying a model to a serving layer using serverless functions

The tutorial is divided into four parts, each with a dedicated Jupyter notebook.
The notebooks are designed to be run sequentially, as each notebook relies on the execution of the previous notebook:

```{toctree}
:maxdepth: 2

01-mlrun-basics
02-model-training
03-model-serving
04-pipeline.ipynb
```
