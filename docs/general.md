<a id="top"></a>
# General Concept and Motivation <!-- omit in toc -->
- [The Challenge](#the-challenge)
- [The MLRun Vision](#the-mlrun-vision)

## The Challenge

As an ML developer or data scientist, you typically want to write code in your preferred local development environment (IDE) or web notebook, and then run the same code on a larger cluster using scale-out containers or functions.
When you determine that the code is ready, you or someone else need to transfer the code to an automated ML workflow (for example, using [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-quickstart/)).
This pipeline should be secure and include capabilities such as logging and monitoring, as well as allow adjustments to relevant components and easy redeployment.

However, the implementation is challenging: various environments (**"runtimes"**) use different configurations, parameters, and data sources.
In addition, multiple frameworks and platforms are used to focus on different stages of the development life cycle.
This leads to constant development and DevOps/MLOps work.

Furthermore, as your project scales, you need greater computation power or GPUs, and you need to access large-scale data sets.
This cannot work on laptops.
You need a way to seamlessly run your code on a remote cluster and automatically scale it out.

## The MLRun Vision

When ML running experiments, you should ideally be able to record and version your code, configuration, outputs, and associated inputs (lineage), so you can easily reproduce and explain your results.
The fact that you probably need to use different types of storage (such as files and AWS S3 buckets) and various databases, further complicates the implementation.

Wouldn't it be great if you could write the code once, using your preferred development environment and simple "local" semantics, and then run it as-is on different platforms?
Imagine a layer that automates the build process, execution, data movement, scaling, versioning, parameterization, outputs tracking, and more.
A world of easily developed, published, or consumed data or ML "functions" that can be used to form complex and large-scale ML pipelines.

In addition, imagine a marketplace of ML functions that includes both open-source templates and your internally developed functions, to support code reuse across projects and companies and thus further accelerate your work.

<b>This is the goal of MLRun.</b>

> **Note:** The code is in early development stages and is provided as a reference.
> The hope is to foster wide industry collaboration and make all the resources pluggable, so that developers can code to a single API and use various open-source projects or commercial products.


[Back to top](#top)
