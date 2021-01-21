<a id="top"></a>
# General Concept and Motivation <!-- omit in toc -->
- [The Challenge](#the-challenge)
- [Why MLRun?](#why-mlrun)
- [Basic Components](#basic-components)

<a id="the-challenge"></a>
## The Challenge

As an ML developer or data scientist, you typically want to write code in your preferred local development environment (IDE) or web notebook, and then run the same code on a larger cluster using scale-out containers or functions.
When you determine that the code is ready, you or someone else need to transfer the code to an automated ML workflow (for example, using [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-quickstart/)).
This pipeline should be secure and include capabilities such as logging and monitoring, as well as allow adjustments to relevant components and easy redeployment.

However, the implementation is challenging: various environments ("runtimes") use different configurations, parameters, and data sources.
In addition, multiple frameworks and platforms are used to focus on different stages of the development life cycle.
This leads to constant development and DevOps/MLOps work.

Furthermore, as your project scales, you need greater computation power or GPUs, and you need to access large-scale data sets.
This cannot work on laptops.
You need a way to seamlessly run your code on a remote cluster and automatically scale it out.

<a id="why-mlrun"></a>
## Why MLRun?

When running ML experiments, you should ideally be able to record and version your code, configuration, outputs, and associated inputs (lineage), so you can easily reproduce and explain your results.
The fact that you probably need to use different types of storage (such as files and AWS S3 buckets) and various databases, further complicates the implementation.

Wouldn't it be great if you could write the code once, using your preferred development environment and simple "local" semantics, and then run it as-is on different platforms?
Imagine a layer that automates the build process, execution, data movement, scaling, versioning, parameterization, outputs tracking, and more.
A world of easily developed, published, or consumed data or ML "functions" that can be used to form complex and large-scale ML pipelines.

In addition, imagine a marketplace of ML functions that includes both open-source templates and your internally developed functions, to support code reuse across projects and companies and thus further accelerate your work.

<b>This is the goal of MLRun.</b>

> **Note:** The code is in early development stages and is provided as a reference.
> The hope is to foster wide industry collaboration and make all the resources pluggable, so that developers can code to a single API and use various open-source projects or commercial products.

[Back to top](#top)

<a id="basic-components"></a>
## Basic Components

MLRun has the following main components:

- <a id="def-project"></a>**Project** &mdash; a container for organizing all of your work on a particular activity.
    Projects consist of metadata, source code, workflows, data and artifacts, models, triggers, and member management for user collaboration.

- <a id="def-function"></a>**Function** &mdash; a software package with one or more methods and runtime-specific attributes (such as image, command, arguments, and environment).

- <a id="def-run"></a>**Run** &mdash; an object that contains information about an executed function.
    The run object is created as a result of running a function, and contains the function attributes (such as arguments, inputs, and outputs), as well the execution status and results (including links to output artifacts).

- <a id="def-artifact"></a>**Artifact** &mdash; versioned data artifacts (such as data sets, files and models) that are produced or consumed by functions, runs, and workflows.

- <a id="def-workflow"></a>**Workflow** &mdash; defines a functions pipeline or a directed acyclic graph (DAG) to execute using [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-quickstart/).

- <a id="def-ui"></a>**UI** &mdash; a graphical user interface (dashboard) for displaying and managing projects and their contained experiments, artifacts, and code.

[Back to top](#top)
