# Deployment and monitoring

  
## Real-time Pipelines

MLRun serving graphs allow to easily build real-time pipelines that include data processing, advanced model serving, custom logic, and fast access to a variety of data systems, and deploy them quickly to production with minimal effort.

High-level transformation logic is automatically converted to real-time serverless processing engines that can accept events or online data, handle any type of structured or unstructured data, and run complex computation graphs and native user code. The processing engines can access Iguazio’s [real-time multi-model database](https://www.iguazio.com/docs/latest-release/data-layer/), to retrieve and manipulate state and data at scale.

MLRun graph capabilities include:

- Easy to build and deploy distributed real-time computation graphs
- Use the real-time serverless engine (Nuclio) for auto-scaling and optimized resource utilization
- Built-in operators to handle data manipulation, IO, machine learning, deep-learning, NLP, etc.
- Built-in monitoring for performance, resources, errors, data, model behaviour, and custom metrics
- Debug in the IDE/Notebook, deploy to production using a single command

The serving graphs are used by [MLRun’s Feature Store](../feature-store/feature-store) to build real-time feature engineering pipelines, and are used to deploy and serve ML/DL models (read more about [model serving using the graphs](../serving/build-graph-model-serving)).

**Accelerate performance and time to production**

The underlying Nuclio serverless engine uses a high-performance parallel processing engine that maximizes the 
utilization of CPUs and GPUs, supports 13 protocols and invocation methods (for example, HTTP, Cron, Kafka, Kinesis), 
and includes dynamic auto-scaling for HTTP and streaming. Nuclio and MLRun support the full life cycle, including auto-
generation of micro-services, APIs, load-balancing, logging, monitoring, and configuration management&mdash;such that 
developers can focus on code, and deploy to production faster with minimal work.

## Model Serving

MLRun Serving allow composition of multi-stage real-time pipelines made of serverless Nuclio functions, including data processing, 
advanced model serving, custom logic, and fast access to a variety of data systems, and deploying them quickly to production with 
minimal effort.

High-level transformation logic is automatically converted to real-time serverless processing engines that can accept events or online data, 
handle any type of structured or unstructured data, and run complex computation graphs and native user code. The processing engines can 
access [Iguazio’s real-time multi-model database](https://www.iguazio.com/docs/latest-release/data-layer/), to retrieve and manipulate 
state and data at scale.

MLRun serving graph capabilities include:
- Easy to build and deploy distributed real-time computation graphs.
- Use the real-time serverless engine (Nuclio) for auto-scaling and optimizing resource utilization.
- Built-in operators to handle data manipulation, IO, machine learning, deep-learning, NLP, etc.
- Built-in monitoring for performance, resources, errors, data, model behavior, and custom metrics.
- Debug in the IDE/Notebook, deploy to production using a single command.

The serving graphs are used by [MLRun's Feature Store](../feature-store/feature-store.md) to build real-time feature engineering pipelines 
([Real-time serving pipelines](../serving/serving-grap)), 
and are used to deploy and serve ML/DL models. Read more about [model serving](../build-graph-model-serving.md) using the graphs.


## Model Monitoring

Model performance monitoring is a basic operational task that is implemented after an AI model has been deployed. MLRun:
- Monitors your models in production, and identifies and mitigates drift on the fly.
- Detects model drift based on feature drift via the integrated feature store, and auto-triggers retraining.

See full details in [Model monitoring](../model_monitoring/index).


## CI/CD and Automation

You can run your ML Pipelines using CI frameworks like Github Actions, GitLab CI/CD, etc. MLRun supports a simple and native integration 
with the CI systems. 

- Build/run complex workflows composed of local/library functions or external cloud services (e.g. AutoML)
- Support various Pipeline/CI engines (Kubeflow, GitHub, Gitlab, Jenkins)
- Track & version code, data, params, results with minimal effort
- Elastic scaling of each step
- Extensive function marketplace 


See full details in [Github/Gitlab and CI/CD Integration](../projects/ci-integration)
