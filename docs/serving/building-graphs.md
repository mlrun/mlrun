(building-graphs)=
# Building graphs

MLRun comes with pre-built steps that include data manipulation, readers, writers, and model serving.
A step runs a function, class handler, or a REST API call.
A step can also be an external REST API (the special `$remote` class).
You can also write your own steps using standard Python functions or custom functions/classes.

**In this section**

```{toctree}
:maxdepth: 1
basic-steps
model-serving-steps
remote-steps
hub-steps
writing-custom-steps
router-steps
router
```
**See also**
- [Queues and streams](../serving/remote-execution.ipynb#queues-and-streams)
- {ref}`Importing steps from the MLRun hub<load-from-hub>`