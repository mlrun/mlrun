(serving-function)=
# Function of type `serving`

Deploying models in MLRun uses the function type `serving`. You can create a serving function using the `set_function()` call from a notebook. 
You can also import an existing serving function/template from the {ref}`function-hub`.

This example converts a notebook to a serving function, adds a model to it, and deploys it:

```python
serving = project.set_function(name="my-serving", func="my_serving.ipynb", kind="serving", image="mlrun/mlrun", handler="handler")
serving.add_model(key="iris", model_path="https://s3.wasabisys.com/iguazio/models/iris/model.pkl", model_class="ClassifierModel")
project.deploy_function(serving)
```


**See also**
- {ref}`Real-time serving pipelines (graphs) <serving-graph>`: higher level real-time graphs (DAG) over one or more Nuclio functions
- {ref}`Serving graphs demos and tutorials <demos-serving>` 
- {ref}`Real-time serving <mlrun-serving-overview>`
- {ref}`Serving pre-trained ML/DL models <serving-ml-dl-models>`

