(router-steps)=
# Router steps

Router steps allow branching, enrichment, aggregation, etc.

**In this section**
- [VotingEnsemble](#votingensemble)
- [Parallel execution](#parallel-execution-parallelrun)

**See also**
- Example of {py:class}`~mlrun.serving.RouterStep` in {ref}`graph-example`.

```{admonition} Note
The `*` prefix indicates a router class (not a simple processing step).
```

## Router
### Description
{py:class}`~mlrun.serving.RouterStep' implements routing logic for running child routes.

### Use Case


### Examples

## VotingEnsemble
### Description
{py:class}`~mlrun.serving.routers.VotingEnsemble` is a router that encapsulates both execution and aggregation of multiple model routes. It outputs a single result.

### Use Case


### Examples
```python
ensemble = graph.add_step(
    "*mlrun.serving.VotingEnsemble", name="ensemble", vote_type="regression"
)  # for numeric output
```

In classification mode:

```python
ensemble = graph.add_step(
    "*mlrun.serving.VotingEnsemble", name="ensemble", vote_type="classification"
)  # for categorical output
```

A full flow example:

```python
import mlrun

project_name = "flow-with-routes"
project = mlrun.get_or_create_project(
    project_name, context="./", allow_cross_project=True
)

fn = mlrun.code_to_function(
    name="routers_example",
    kind="serving",
    project=project_name,
    filename="serving_code.py",
    image="mlrun/mlrun",
)

graph = fn.set_topology("flow")

# 1. Preprocessing
graph.add_step("PreProcess", class_name="PreProcessClass")

# 2. Voting
models_path = "https://s3.wasabisys.com/iguazio/models/iris/model.pkl"
path1 = models_path
path2 = models_path

router = graph.add_step(
    "*mlrun.serving.VotingEnsemble",
    name="ensemble",
    vote_type="classification",
)
router.add_route("m1", class_name="ClassifierModel", model_path=path1)
router.add_route("m2", class_name="ClassifierModel", model_path=path2)

# 3. Postprocess
graph.add_step("PostProcess", class_name="PostProcessClass", after="enricher").respond()

# 4. Local test
server = fn.to_mock_server()
result = server.test(body={"user_id": 123, "inputs": [0.1, 0.2, 0.3]})
print(result)
```

---
Routers are added as:

```python
graph.add_step("*mlrun.serving.ParallelRun", name="parallel")
```



## Parallel execution (ParallelRun)
### Description
Use {py:meth}`~mlrun.serving.routers.ParallelRun` to run multiple independent branches with custom merging or postprocessing. It outputs a dict or list.

## Use Cases


### Examples

```python
parallel = graph.add_step("*mlrun.serving.ParallelRun", name="parallel_models")
parallel.add_route("model_a", class_name="Cls1")
```

Downstream, you can add custom merger or processing steps.