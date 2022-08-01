(build-graph-model-serving)=
# Model serving pipelines

MLRun serving can produce managed real-time serverless pipelines from various tasks, including MLRun models or standard model files. MLRun serving supports complex and distributed graphs (see the [Distributed (Multi-function) Pipeline Example](./distributed-graph.html)), which can involve streaming, data/document/image processing, NLP, and model monitoring, etc. The pipelines use the [Nuclio](https://nuclio.io/) real-time serverless engine, which can be deployed anywhere. 

Simple model serving classes can be written in Python or be taken from a set of pre-developed ML/DL classes. The code can handle complex data, feature preparation, and binary data (such as images and video files). The Nuclio serving engine supports the full model-serving life cycle, including auto-generation of microservices, APIs, load balancing, model logging and monitoring, and configuration management.


