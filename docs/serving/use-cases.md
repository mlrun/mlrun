(use-cases-serving)=
# Use cases

<!-- ## Data preparation, ## Model serving -->

**In this section**
* [Data and feature engineering](#Data-and-feature-engineering-using-the-feature-store)
* [Example of Simple model serving router](#example-of-a-simple-model-serving-router)
* [Example of Advanced data processing and serving ensemble](#example-of-advanced-data-processing-and-serving-ensemble)
* [Example of NLP processing pipeline with real-time streaming](#example-of-nlp-processing-pipeline-with-real-time-streaming)

In addition to the examples in this section, see the:
- [Distributed (multi-function) pipeline example](./distributed-graph.html) that details how to run a pipeline that consists of multiple serverless functions (connected using streams).
- [Advanced Model Serving Graph Notebook Example](./graph-example.html) that illustrates the flow, task, model, and ensemble router states; building tasks from custom handlers; classes and storey components; using custom error handlers; testing graphs locally; deploying a graph as a real-time serverless function.
- [MLRun demos repository](https://github.com/mlrun/demos) for additional use cases and full end-to-end examples, including fraud prevention using the Iguazio feature store, a mask detection demo, and converting existing ML code to an MLRun project.

## Data and feature engineering (using the feature store)

You can build a feature set transformation using serving graphs.

High-level transformation logic is automatically converted to real-time serverless processing engines that can read 
from any online or offline source, handle any type of structures or unstructured data, run complex computation graphs 
and native user code. Iguazio’s solution uses a unique multi-model database, serving the computed features consistently 
through many different APIs and formats (like files, SQL queries, pandas, real-time REST APIs, time-series, streaming), 
resulting in better accuracy and simpler integration.

Read more in {ref}`feature-store`, and [Feature set transformations](../feature-store/transformations.html).

## Example of a simple model serving router

Graphs are used for serving models with different transformations.

To deploy a serving function, you need to import or create the serving function, 
add models to it, and then deploy it.  

```python
    import mlrun  
    # load the sklearn model serving function and add models to it  
    fn = mlrun.import_function('hub://v2_model_server')
    fn.add_model("model1", model_path={model1-url})
    fn.add_model("model2", model_path={model2-url})

    # deploy the function to the cluster
    fn.deploy()
    
    # test the live model endpoint
    fn.invoke('/v2/models/model1/infer', body={"inputs": [5]})
```

The Serving function supports the same protocol used in KFServing V2 and Triton Serving framework. 
To invoke the model, to use following url: `<function-host>/v2/models/model1/infer`.

See the [**serving protocol specification**](./model-api.html) for details.

```{note}
Model url is either an MLRun model store object (starts with `store://`) or URL of a model directory 
(in NFS, s3, v3io, azure, for example `s3://{bucket}/{model-dir}`). Note that credentials might need to 
be added to the serving function via environment variables or MLRun secrets.
```

See the [**scikit-learn classifier example**](https://github.com/mlrun/functions/blob/master/sklearn_classifier/sklearn_classifier.ipynb), 
which explains how to create/log MLRun models.

### Writing your own serving class

You can implement your own model serving or data processing classes. All you need to do is:

1. Inherit the base model serving class.
2. Add your implementation for model `load()` (download the model file(s) and load the model into memory). 
2. `predict()` (accept the request payload and return the prediction/inference results).

You can override additional methods: `preprocess`, `validate`, `postprocess`, `explain`.<br>
You can add custom API endpoints by adding the method `op_xx(event)` (which can be invoked by
calling the `<model-url>/xx`, where operation = xx). See [model class API](https://docs.mlrun.org/en/latest/api/mlrun.model.html).

For an example of writing the minimal serving functions, see [Minimal sklearn serving function example](./custom-model-serving-class.html#minimal-sklearn-serving-function-example).

See the full [V2 Model Server (SKLearn) example](https://github.com/mlrun/functions/blob/master/v2_model_server/v2_model_server.ipynb) that 
tests one or more classifier models against a held-out dataset.

## Example of advanced data processing and serving ensemble

MLRun Serving graphs can host advanced pipelines that handle event/data processing, ML functionality, 
 or any custom task. The following example demonstrates an asynchronous pipeline that pre-processes data, 
passes the data into a model ensemble, and finishes off with post processing. 

**For a complete example, see the [Advanced graph example notebook](./graph-example.html).**

Create a new function of type serving from code and set the graph topology to `async flow`.

```python
import mlrun
function = mlrun.code_to_function("advanced", filename="demo.py", 
                                  kind="serving", image="mlrun/mlrun",
                                  requirements=['storey'])
graph = function.set_topology("flow", engine="async")
```

Build and connect the graph (DAG) using the custom function and classes and plot the result. 
Add steps using the `step.to()` method (adds a new step after the current one), or using the 
`graph.add_step()` method.

If you want the error from the graph or the step to be fed into a specific step (catcher), 
use the `graph.error_handler()` (apply to all steps) or `step.error_handler()` 
(apply to a specific step).

Specify which step is the responder (returns the HTTP response) using the `step.respond()` method. 
If the responder is not specified, the graph is non-blocking.

```python
# use built-in storey class or our custom Echo class to create and link Task steps
graph.to("storey.Extend", name="enrich", _fn='({"tag": "something"})') \
     .to(class_name="Echo", name="pre-process", some_arg='abc').error_handler("catcher")

# add an Ensemble router with two child models (routes), the "*" prefix mark it is a router class
router = graph.add_step("*mlrun.serving.VotingEnsemble", name="ensemble", after="pre-process")
router.add_route("m1", class_name="ClassifierModel", model_path=path1)
router.add_route("m2", class_name="ClassifierModel", model_path=path2)

# add the final step (after the router) which handles post processing and respond to the client
graph.add_step(class_name="Echo", name="final", after="ensemble").respond()

# add error handling step, run only when/if the "pre-process" step fail (keep after="")  
graph.add_step(handler="error_catcher", name="catcher", full_event=True, after="")

# plot the graph (using Graphviz) and run a test
graph.plot(rankdir='LR')
```

<br><img src="../_static/images/graph-flow.svg" alt="graph-flow" width="800"/><br>

Create a mock (test) server, and run a test. Use `wait_for_completion()` 
to wait for the async event loop to complete.
  
```python
server = function.to_mock_server()
resp = server.test("/v2/models/m2/infer", body={"inputs": data})
server.wait_for_completion()
``` 

And deploy the graph as a real-time Nuclio serverless function with one command:

    function.deploy()

```{note}
If you test a Nuclio function that has a serving graph with the async engine via the Nuclio UI, the UI might not display the logs in the output.
```

## Example of an NLP processing pipeline with real-time streaming 

In some cases it's useful to split your processing to multiple functions and use 
streaming protocols to connect those functions. In this example the data 
processing is in the first function/container and the NLP processing is in the second function. 
In this example the GPU contained in the second function.

See the [full notebook example](./distributed-graph.html).

```python
# define a new real-time serving function (from code) with an async graph
fn = mlrun.code_to_function("multi-func", filename="./data_prep.py", kind="serving", image='mlrun/mlrun')
graph = fn.set_topology("flow", engine="async")

# define the graph steps (DAG)
graph.to(name="load_url", handler="load_url")\
     .to(name="to_paragraphs", handler="to_paragraphs")\
     .to("storey.FlatMap", "flatten_paragraphs", _fn="(event)")\
     .to(">>", "q1", path=internal_stream)\
     .to(name="nlp", class_name="ApplyNLP", function="enrich")\
     .to(name="extract_entities", handler="extract_entities", function="enrich")\
     .to(name="enrich_entities", handler="enrich_entities", function="enrich")\
     .to("storey.FlatMap", "flatten_entities", _fn="(event)", function="enrich")\
     .to(name="printer", handler="myprint", function="enrich")\
     .to(">>", "output_stream", path=out_stream)

# specify the "enrich" child function, add extra package requirements
child = fn.add_child_function('enrich', './nlp.py', 'mlrun/mlrun')
child.spec.build.commands = ["python -m pip install spacy",
                             "python -m spacy download en_core_web_sm"]
graph.plot()
```

Currently queues support iguazio v3io and Kafka streams.
