(serving-graph)=
# Real-time serving pipelines (graphs)

MLRun graphs, both cyclic graphs and DAGs (directed acyclic graph), are easy to build and deploy.
Graphs are composed of individual steps. The serving graphs can be composed of {ref}`pre-defined graph steps<building-graphs>`,  [custom steps](./writing-custom-steps.ipynb), from native python classes/functions, and step you import from the MLRun hub or your own hub. 

Serving graph features include:
- Cyclic graphs for use in iterative and agentic workflows, enabling patterns like the evaluator–optimizer loop, guardrail enforcement, large-scale data or inference pipelines, and multi-agent communication. Typical use cases include agent steps that require feedback, retry, or coordination loops (common in GenAI-driven workflows).
- Batching, whereby you can control which parts of graph use batching. 
- Streaming responses whereby tokens are sent back as they are generated rather than wait until all the tokens are generated. You can use multiple streaming steps in a graph. Streaming responses reduce latency.
- The [ModelRunnerStep](../serving/model-serving-steps.md#modelrunnerstep), for running multiple models on each event with control over how they are executed in terms of concurrency and parallelism. 
- LLM support. When a ModelRunnerStep is included in a graph, MLRun automatically imports the default language model class during function deployment to wrap the model for handling an LLM prompt-based inference. See an example in {ref}`genai-serving-graph`.

Graphs can run inside your IDE or Notebook for test and simulation. 

Serving graphs are built on top of [Nuclio](https://docs.nuclio.io/en/latest/) (real-time serverless engine), [MLRun jobs](../concepts/scheduled-jobs.md), [MLRun Storey](<https://github.com/mlrun/storey>) (native Python async and stream processing engine), and other MLRun facilities. 

By default, all steps of the serving graph run on the same pod. It is possible to run different steps on different pods using distributed pipelines.Typically you run steps that require CPU on one pod, and steps that require a GPU on a different pod that is running on a potentially different node that has GPU support.

The realtime pipelines UI page displays: 
- A table of serving graphs with a few parameters that can be filtered
- The total number of graphs/pipelines, the status of the main function, and the total number of endpoints
- Each graph step is identified by an icon according to its category, and displays details of the graph steps
- A model endpoints tab

**In this section**

```{toctree}
:maxdepth: 1
basic-example
getting-started
building-graphs
deploying-graphs
demos
graph-advanced-cfg
```