# Deployment and monitoring

**In this section**
- [Real-time pipelines](#real-time-pipelines)
- [Model serving](#model-serving)
- [Model monitoring](#model-monitoring)
- [CI/CD and automation](#ci-cd-and-automation)

## Real-time pipelines

MLRun graphs enable building and running DAGs (directed acyclic graphs). Graphs are composed of individual steps. The first graph element accepts an Event object, transforms/processes the event and passes the result to the next step in the graph. The final result can be written out to some destination (file, DB, stream, etc.) or returned back to the caller (one of the graph steps can be marked with `.respond()`).

MLRun graph capabilities include:

- Easy to build and deploy distributed real-time computation graphs
- Use the real-time serverless engine (Nuclio) for auto-scaling and optimized resource utilization
- Built-in operators to handle data manipulation, IO, machine learning, deep-learning, NLP, etc.
- Built-in monitoring for performance, resources, errors, data, model behaviour, and custom metrics
- Debug in the IDE/Notebook

The serving graphs are used by [MLRun’s Feature Store](../feature-store/feature-store.html) to build real-time feature engineering pipelines. 

See full details and examples in [Real-time serving pipelines (graphs)](../serving/serving-graph.html).

## Model serving

MLRun Serving allow composition of multi-stage real-time pipelines made of serverless Nuclio functions, including data processing, 
advanced model serving, custom logic, and fast access to a variety of data systems, and deploying them quickly to production with 
minimal effort.

High-level transformation logic is automatically converted to real-time serverless processing engines that can accept events or online data, 
handle any type of structured or unstructured data, and run complex computation graphs and native user code. 

Graphs are used to deploy and serve ML/DL models. Graphs can be deployed into a production serverless pipeline with a single command. 

See full details and examples in [model serving using the graphs](../serving/build-graph-model-serving.html).


## Model monitoring

Model performance monitoring is a basic operational task that is implemented after an AI model has been deployed. MLRun:
- Monitors your models in production, and identifies and mitigates drift on the fly.
- Detects model drift based on feature drift via the integrated feature store, and auto-triggers retraining.

See full details and examples in [Model monitoring](../model_monitoring/index.html).


## CI/CD and automation

You can run your ML Pipelines using CI frameworks like Github Actions, GitLab CI/CD, etc. MLRun supports a simple and native integration 
with the CI systems. 

- Build/run complex workflows composed of local/library functions or external cloud services (e.g. AutoML)
- Support various Pipeline/CI engines (Kubeflow, GitHub, Gitlab, Jenkins)
- Track & version code, data, params, results with minimal effort
- Elastic scaling of each step
- Extensive function marketplace 


See full details and examples in [Github/Gitlab and CI/CD Integration](../projects/ci-integration.html).
