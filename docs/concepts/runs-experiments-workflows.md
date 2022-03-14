# Runs, experiments and workflows

## Workflows

A workflow is a definition of execution of functions. It defines the order of execution of multiple dependent steps in a DAG. A workflow 
can reference the project’s params, secrets, artifacts, etc. It can also use a function execution output as a function execution 
input (which, of course, defines the order of execution).

MLRun supports running workflows on a `local` or [`kubeflow`](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/) pipeline engine. The `local` engine runs the workflow as a 
local process, which is simpler for debuggimg and running simple/sequential tasks. The `kubeflow` ("kfp") engine runs as a task over the 
cluster and supports more advanced operations (conditions, branches, etc.). You can select the engine at runtime. Kubeflow-specific
directives like conditions and branches are not supported by the `local` engine.

Workflows are saved/registered in the project using the {py:meth}`~mlrun.projects.MlrunProject.set_workflow`.  
Workflows are executed using the {py:meth}`~mlrun.projects.MlrunProject.run` method or using the CLI command `mlrun project`.

See full details in [Project Workflows and Automation](../projects/workflows).


## The run context, logging

After running a job, you need to be able to track it, including viewing the run parameters, inputs, and outputs. MLRun uses a runtime "context" object inside the code. This provides access to parameters, data, secrets, etc., as well as log text, files, artifacts, and labels.

- If `context` is specified as the first parameter in the function signature, MLRun injects the current job context into it.
- Alternatively, you can obtain the context object using the MLRun `get_or_create_ctx()` method, without changing the function.

You can define the code to get parameters and inputs from the context, as well as log run outputs, artifacts, tags, and 
time-series metrics in the context.


## Auto logging & MLOps 


```{toctree}
:maxdepth: 2
  
../runtimes/distributed
```