(mlrun-ce-overview)=
# MLRun CE Overview

This page provides a comprehensive overview of MLRun Community Edition, the open-source MLOps platform for building and managing AI projects.

**In this section**
- [What is MLRun CE](#what-is-mlrun-ce)
- [Main Advantages](#mlrun-ce-main-advantages)
- [Ecosystem](#ecosystem)
- [Core Components](#core-components)
- [Storage Resources](#storage-resources)
- [Addional References](#addional-references)

## What is MLRun CE

MLRun CE is an open-source platform that simplifies the entire lifecycle of your AI project. 
By installing the MLRun CE Helm chart on your Kubernetes cluster or local laptop, you get a powerful, integrated environment for development your LLM and ML projects. 
The platform is built on two main applications - MLRun for MLOps orchestration and Nuclio for serverless computing.

### **MLRun: The MLOps Orchestration Framework**
MLRun is the MLOps orchestration framework that automates the entire AI pipeline, from data preparation and model training to deployment and management. It automates tasks like model tuning and optimization, enabling you to build scalable and observable AI applications. With MLRun, you can run your batch jobs and your real-time applications over elastic resources and gain end-to-end observability.

###  **Nuclio: The Serverless Engine**
Nuclio is a high-performance serverless framework that focuses on data, I/O, and compute-intensive workloads. It is the engine that powers the real-time functions within MLRun. Nuclio allows you to deploy your code as serverless functions, which are highly efficient and can process hundreds of thousands of events per second. It supports various data sources, triggers, and execution over CPUs and GPUs.

## MLRun CE Main Advantages 

1. **Open-source MLOps Solution** – MLRun CE is an open-source MLOps platform that you can quickly install on your Kubernetes cluster or local desktop by deploying the mlrun-ce chart.

2. **Rapid Project Development** - Allows you to take your code from a Jupyter Notebook or you local IDE to a scalable k8s based platform, with minimal changes. 
This significantly shortens the time-to-production, enabling faster iteration and business 

3. **Efficient AI Project Management** – Gives users tools for experiment tracking, hyperparameter tuning, and model selection, allowing you to easily compare experiments, optimize models, and ensure reproducibility.

4. **Scalability and Efficiency** - Automatically and elastically scale resources based on demand. This ensures your workloads, whether it is batch or real-time, run efficiently, reducing computation costs. It's particularly useful for resource-intensive tasks like LLM fine-tuning or inference.

5. **MLRun Model Monitoring** – Features a comprehensive model monitoring solution that lets users track their models, compare results and performance metrics, and detect data drift or anomalous behavior. 
It also supports automated alerts for model exceptions, enabling proactive maintenance and ensuring continued model reliability.

6. **Seamless Integrations** – MLRun CE seamlessly connects with a broad ecosystem of leading open-source tools—including Kubeflow Pipelines (KFP) for workflow orchestration, Spark for large-scale data processing, and Grafana for interactive visualization. Its flexible, open architecture enables you to incorporate your preferred tools and workflows, accelerating adoption and productivity.

## Ecosystem
<p align="center"><img src="../_static/images/mlrun-ce-diagram.jpg" alt="MLRun CE Ecosystem" width="800"/></p><br>

## Core Components
* MLRun - https://github.com/mlrun/mlrun
  - MLRun API
  - MLRun UI
  - MLRun DB (MySQL)
* Nuclio - https://github.com/nuclio/nuclio
* Jupyter - https://github.com/jupyter/notebook (+MLRun integrated)
* MPI Operator - https://github.com/kubeflow/mpi-operator
* MinIO - https://github.com/minio/minio/tree/master/helm/minio
* Spark Operator - https://github.com/GoogleCloudPlatform/spark-on-k8s-operator
* Prometheus stack - https://github.com/prometheus-community/helm-charts
  - Prometheus
  - Grafana
* MLRun Model Monitoring - 
  - Kafka - https://github.com/bitnami/charts/tree/main/bitnami/kafka
  - TDengine - https://github.com/taosdata/TDengine-Operator/blob/3.0/helm/tdengine/values.yaml
* KFP Pipelines - https://github.com/kubeflow/pipelines

## **Storage resources**
When installing the MLRun Community Edition, several storage resources are created:

- **PVs via default configured storage class**: Holds the file system of the stacks pods, including the MySQL database of MLRun, MinIO for artifacts and Pipelines Storage and more. 
These are not deleted when the stack is uninstalled, which allows upgrading without losing data.

   See also MLRun data store [documentation](https://docs.mlrun.org/en/stable/store/datastore.html)

- **Container Images in the configured docker-registry**: When building and deploying MLRun and Nuclio functions via the MLRun Community Edition, the function images are 
stored in the given configured docker registry. These images persist in the docker registry and are not deleted.

## Addional References 

- **Documentation:** [MLRun Docs](https://docs.mlrun.org) | [Nuclio Docs](https://docs.nuclio.io/en/latest/index.html)
- **Quick Start:** [MLRun basics tutorial](https://docs.mlrun.org/en/stable/tutorials/01-mlrun-basics.html)
- **Cheat Sheet:** [MLRun cheat sheet](https://docs.mlrun.org/en/stable/cheat-sheet.html)
- **Community:** [Join our Slack](https://mlopslive.slack.com) for support and discussions
- **GitHub:** [MLRun Repository](https://github.com/mlrun/mlrun)

