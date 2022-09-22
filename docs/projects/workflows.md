(workflows)=
# Batch runs and workflows

A workflow is a definition of execution of functions. It defines the order of execution of multiple dependent steps in a  directed acyclic graph (DAG). A workflow 
can reference the project’s params, secrets, artifacts, etc. It can also use a function execution output as a function execution 
input (which, of course, defines the order of execution).

MLRun supports running workflows on a `local` or [`kubeflow`](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/) pipeline engine. The `local` engine runs the workflow as a 
local process, which is simpler for debugging and running simple/sequential tasks. The `kubeflow` ("kfp") engine runs as a task over the 
cluster and supports more advanced operations (conditions, branches, etc.). You can select the engine at runtime. Kubeflow-specific
directives like conditions and branches are not supported by the `local` engine.

Workflows are saved/registered in the project using the {py:meth}`~mlrun.projects.MlrunProject.set_workflow`.  
Workflows are executed using the {py:meth}`~mlrun.projects.MlrunProject.run` method or using the CLI command `mlrun project`.

Refer to the **{ref}`tutorial`** for complete examples.

**In this section**

```{toctree}
:maxdepth: 1

../concepts/workflow-overview
../concepts/mlrun-execution-context
../concepts/submitting-tasks-jobs-to-functions
<!-- ../runtimes/run-multistage-workflows -->
../concepts/scheduled-jobs
../concepts/model-training
```