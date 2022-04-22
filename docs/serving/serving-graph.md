(serving)=

# Real-time serving pipelines (graphs)

MLRun graphs enable building and running DAGs (directed acyclic graph). Graphs are composed of individual steps. 
The first graph element accepts an `Event` object, transform/process the event and pass the result to the next steps
in the graph. The final result can be written out to some destination (file, DB, stream, etc.) or returned back to the caller
(one of the graph steps can be marked with `.respond()`). 

The serving graphs can be composed of [pre-defined graph steps](./available-steps.md), block-type elements (model servers, routers, ensembles, 
data readers and writers, data engineering tasks, validators, etc.), [custom steps](./writing-custom-steps.ipynb), or from native python 
classes/functions. A graph can have data processing steps, model ensembles, model servers, post-processing, etc. (see the [Advanced Model Serving Graph Notebook Example](./graph-example.ipynb)). Graphs can auto-scale and span multiple function containers (connected through streaming protocols).

![serving graph high level](../_static/images/serving-graph-high-level.png)
  
Different steps can run on the same local function, or run on a remote function. You can call existing functions from the graph and reuse them from other graphs, as well as scale up and down different components individually.

Graphs can run inside your IDE or Notebook for test and simulation. Serving graphs are built on 
top of [Nuclio](https://github.com/nuclio/nuclio) (real-time serverless engine), MLRun Jobs, 
[MLRun Storey](<https://github.com/mlrun/storey>) (native Python async and stream processing engine), 
and other MLRun facilities. 

**In this section**

```{toctree}
:maxdepth: 2
  
getting-started
use-cases
realtime-pipelines
writing-custom-steps
available-steps
<!--- best-practice --->
demos
graph-ha-cfg
```
