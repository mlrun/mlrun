.. _serving:

Real-time serving pipelines (graphs)
--------------------------------------------------

MLRun serving graphs allow to easily build real-time data processing and 
advanced model serving pipelines, and deploy them quickly to production with
minimal effort.

High-level transformation logic is automatically converted to real-time serverless processing engines that can read 
from any online or offline source, handle any type of structured or unstructured data, and run complex computation graphs 
and native user code. Iguazio’s solution uses a unique multi-model database, serving the computed features consistently through many different APIs and formats (like files, SQL queries, pandas, real-time REST APIs, time-series, streaming), resulting in better accuracy and simpler integration.

MLRun graph capabilities include:

- Easy to build and manage complex graphs
- Real-time serverless engine for optimized utilization (auto scale up/down)
- Advanced applications including streaming, machine learning, deep-learning, NLP
- Debug in the IDE/Notebook, deploy to production using a single command

**Accelerate performance and time to production**


The underlying Nuclio serverless engine uses a high-performance parallel processing 
engine that maximizes the utilization of CPUs and GPUs, supports 13 protocols and 
invocation methods (for example, HTTP, Cron, Kafka, Kinesis), and includes dynamic auto-scaling for 
HTTP and streaming. Nuclio and MLRun support the full life cycle, including auto-
generation of micro-services, APIs, load-balancing, logging, monitoring, and 
configuration management, allowing developers to focus on code, and deploy faster 
to production with minimal work.


.. toctree::
   :maxdepth: 2
  
   getting-started
   use-cases
   realtime-pipelines
   writing-custom-steps
   available-steps
   build-graph-model-serving
   <!-- best-practice -->
   demos
   graph-ha-cfg
  