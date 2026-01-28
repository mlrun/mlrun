(building-graphs)=
# Building graphs

A step runs a function, class handler, or a REST API call: MLRun comes with pre-built steps that include data manipulation, readers, writers, and model serving.
A step can also be an external REST API (the special `$remote` class).
You can also write your own steps using standard Python functions or custom functions/classes.

**In this section**

```{toctree}
:maxdepth: 1
available-steps
model-serving-steps
remote-steps
writing-custom-steps
queue-step
router
```
See also
- {ref}`load-from-hub`