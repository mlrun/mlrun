(tutorial)=
# Tutorials and Examples

The following tutorials provide a hands-on introduction to using MLRun to implement a data science workflow and automate machine-learning operations (MLOps).

Make sure you start with the [**Quick Start Tutorial**](./01-mlrun-basics.html) to understand the basics.

Each of the following tutorials is a dedicated Jupyter notebook. You can download them by clicking the `download` icon at the top of each page.

```{toctree}
:maxdepth: 1

01-mlrun-basics
02-model-training
03-model-serving
04-pipeline
./taxi/apply-mlrun-on-existing-code
../feature-store/basic-demo
```

You can find different end-to-end demos in MLRun demos repository at [**github.com/mlrun/demos**](https://github.com/mlrun/demos).
Alternatively, use the interactive MLRun [**Katacoda Scenarios**](https://www.katacoda.com/mlrun) that teach how to install and use MLRun. See

```{toctree}
:maxdepth: 1

MLRun demos repository <https://github.com/mlrun/demos>
MLRun Katakoda Scenarios <https://www.katacoda.com/mlrun>
```
## Running the demos in Open Source MLRun

By default, these demos work with the online feature store, which is currently not part of the Open Source MLRun default deployment:
- fraud-prevention-feature-store 
- network-operations
- azureml_demo