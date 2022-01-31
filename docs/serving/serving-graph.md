(serving)=

# Real-time serving pipelines (graphs)

MLRun serving graphs allow to easily build real-time pipelines that include data processing, advanced model serving, 
custom logic, and fast access to a variety of data systems, and deploy them quickly to production with minimal effort.

High-level transformation logic is automatically converted to real-time serverless processing engines that can accept events or online data, 
handle any type of structured or unstructured data, and run complex computation graphs and native user code. The processing engines can 
access [Iguazio’s real-time multi-model database](https://www.iguazio.com/docs/latest-release/data-layer/), to retrieve and manipulate 
state and data at scale.

MLRun graph capabilities include:
- Easy to build and deploy distributed real-time computation graphs
- Use the real-time serverless engine (Nuclio) for auto-scaling and optimized resource utilization
- Built-in operators to handle data manipulation, IO, machine learning, deep-learning, NLP, etc.
- Built-in monitoring for performance, resources, errors, data, model behaviour, and custom metrics
- Debug in the IDE/Notebook, deploy to production using a single command

The serving graphs are used by [MLRun's Feature Store](../feature-store/feature-store.md) to build real-time feature engineering pipelines, 
and are used to deploy and serve ML/DL models (read more about [model serving](./build-graph-model-serving.md) using the graphs).

**Accelerate performance and time to production**

The underlying Nuclio serverless engine uses a high-performance parallel processing engine that maximizes the 
utilization of CPUs and GPUs, supports 13 protocols and invocation methods (for example, HTTP, Cron, Kafka, Kinesis), 
and includes dynamic auto-scaling for HTTP and streaming. Nuclio and MLRun support the full life cycle, including auto-
generation of micro-services, APIs, load-balancing, logging, monitoring, and configuration management&mdash;such that 
developers can focus on code, and deploy to production faster with minimal work.

**These sections provide full details to get you started with serving graphs, including examples:**

```{toctree}
:maxdepth: 2
  
getting-started
use-cases
realtime-pipelines
build-graph-model-serving
writing-custom-steps
available-steps
<!-- best-practice -->
demos
graph-ha-cfg
```