(serving-graph)=
# Real-time serving pipelines (graphs)

MLRun graphs enable building and running DAGs (directed acyclic graph) that are easy to build and deploy, including distributed real-time computation graphs; use the real-time serverless engine (Nuclio) for auto-scaling and optimized resource utilization; use built-in operators to handle data manipulation, IO, machine learning, deep-learning, NLP, etc.; use built-in monitoring for performance, resources, errors, data, model behaviour, and custom metrics; can be debugged in the IDE/Notebook.

Graphs are composed of individual steps. 
The first graph element accepts an `Event` object, transforms/processes the event and passes the result to the next steps
in the graph. The final result can be written out to some destination (file, DB, stream, etc.) or returned back to the caller
(one of the graph steps can be marked with `.respond()`). 

Different steps can run on the same local function, or run on a remote function. You can call existing functions from the graph and reuse 
them from other graphs, as well as scale up and down the different components individually.

The serving graphs can be composed of [pre-defined graph steps](./available-steps.md), block-type elements (model servers, routers, ensembles, 
data readers and writers, data engineering tasks, validators, etc.), [custom steps](./writing-custom-steps.ipynb), or from native python 
classes/functions. A graph can have data processing steps, model ensembles, model servers, post-processing, etc. (see the [Advanced Model Serving Graph Notebook Example](./graph-example.ipynb)). Graphs can auto-scale and span multiple function containers (connected through streaming protocols).
  
<img src="../_static/images/serving-graph-high-level.png" height="3cm">

Graphs can run inside your IDE or Notebook for test and simulation. Serving graphs are built on 
top of [Nuclio](https://github.com/nuclio/nuclio) (real-time serverless engine), [MLRun jobs](../concepts/scheduled-jobs.md), 
[MLRun Storey](<https://github.com/mlrun/storey>) (native Python async and stream processing engine), 
and other MLRun facilities. 

The serving graphs are used by [MLRun’s Feature Store](../feature-store/feature-store.md) to build real-time feature engineering pipelines. 

**In this section**

```{toctree}
:maxdepth: 1
getting-started
use-cases
model-serving-get-started
deploying-graphs
demos
graph-advanced-cfg
```
