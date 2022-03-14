# Using graphs for model serving

With MLRun Serving you compose a graph of steps (composed of pre-defined graph blocks or native python classes/functions). A graph can have data processing steps, model ensembles, model servers, post-processing, etc. (see the [Advanced Model Serving Graph Notebook Example](./graph-example.ipynb)). MLRun Serving supports complex and distributed graphs (see the [Distributed (Multi-function) Pipeline Example](./distributed-graph.ipynb)), which may involve streaming, data/document/image processing, NLP, and model monitoring, etc.

**This section describes:**

```{toctree}
:maxdepth: 1
model-serving-get-started
advanced-routing
custom-model-serving-class
model-api
model-best-practices
```
