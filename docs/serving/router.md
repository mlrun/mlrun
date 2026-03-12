(router)=
# Router topology
```{admonition} Note
The router topology will be deprecated in an upcoming release.
```

The `router` topology is a minimal configuration with a single router and one or more child routes/models, used for simple model serving or 
single hop configurations. The basic routing logic is to route to the child routes based on the `event.path`.

With the `router` topology you can specify different machine learning models. Each model has a logical name. This name is used to route to the correct model when calling the serving function.

More advanced or custom routing can be used, for example, the ensemble router sends the event to all child routes in parallel, aggregates the result, and responds.

## Built-in steps

- [ModelRouter](#modelrouter)
- [EnrichmentVotingEnsemble](#enrichmentvotingensemble)
### ModelRouter
Description:  Basic model router, for calling different models per each model path. See {py:class}`~mlrun.serving.routers.ModelRouter`.
### Example

```
from sklearn.datasets import load_iris

# set the topology/router
graph = fn.set_topology("router")

# Add the model
fn.add_model(
    "model1",
    class_name="ClassifierModel",
    model_path="https://s3.wasabisys.com/iguazio/models/iris/model.pkl",
)

# Add additional models
# fn.add_model("model2", class_name="ClassifierModel", model_path="<path2>")

# create and use the graph simulator
server = fn.to_mock_server()
x = load_iris()["data"].tolist()
result = server.test("/v2/models/model1/infer", {"inputs": x})
server.wait_for_completion()

print(result)
```
## EnrichmentVotingEnsemble
### Description
The typical usage is to pass a feature vector URI that points to a registered feature store vector. The router:
- Fetches features from the feature vector  
- Enriches the incoming event  
- Runs the internal model routes  
- Aggregates the results based on `vote_type`

Notes:
- Only `feature_vector_uri` and `vote_type` are required (no need to pass `models` or `feature_store` in many versions).  
- The router internally handles model execution and merging.

```{admonition} Note
The `*` prefix indicates a router class (not a simple processing step).
```

### Use Case


### Example
```python
enricher = graph.add_step(
    "*mlrun.serving.EnrichmentVotingEnsemble",
    name="enricher",
    feature_vector_uri="store://feature-vectors/user_features_v1",
    vote_type="regression",  # or "classification"
)
```


