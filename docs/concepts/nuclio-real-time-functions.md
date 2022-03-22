# Nuclio real-time functions

Nuclio is a high-performance "serverless" framework focused on data, I/O, and compute intensive workloads. It is well integrated with popular 
data science tools, such as Jupyter and Kubeflow; supports a variety of data and streaming sources; and supports execution over CPUs and GPUs. 

You can use Nuclio through a fully managed application service (in the cloud or on-prem) in the Iguazio Data Science Platform. MLRun serving 
utilizes serverless Nuclio functions to create multi-stage real-time pipelines. 

The underlying Nuclio serverless engine uses a high-performance parallel processing engine that maximizes the utilization of CPUs and GPUs, 
supports 13 protocols and invocation methods (for example, HTTP, Cron, Kafka, Kinesis), and includes dynamic auto-scaling for HTTP and 
streaming. Nuclio and MLRun support the full life cycle, including auto-generation of micro-services, APIs, load-balancing, logging, 
monitoring, and configuration management—such that developers can focus on code, and deploy to production faster with minimal work.

Nuclio is extremely fast: a single function instance can process hundreds of thousands of HTTP requests or data records per second. To learn 
more about how Nuclio works, see the Nuclio architecture [documentation](https://nuclio.io/docs/latest/concepts/architecture/). 

Nuclio is secure: Nuclio is integrated with Kaniko to allow a secure and production-ready way of building Docker images at run time.

Read more in the [Nuclio documentation](https://nuclio.io/docs/latest/) and the open-source [MLRun library](https://github.com/mlrun/mlrun).
