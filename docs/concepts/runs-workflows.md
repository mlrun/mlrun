(workflows)=
# Batch runs and workflows

A workflow is a definition of execution of functions: it defines the order of execution of multiple dependent steps in a directed acyclic graph (DAG). A workflow can reference the project’s params, secrets, artifacts, etc. It can also use a 
function execution output as a function execution input (which, of course, defines the order of execution).

MLRun supports running workflows on a three types of engines, see {ref}`local-remote`.

Workflows are saved/registered in the project using the {py:meth}`~mlrun.projects.MlrunProject.set_workflow`.  
Workflows are executed using the {py:meth}`~mlrun.projects.MlrunProject.run` method or using the CLI command `mlrun project`.

See the examples listed below and the **{ref}`tutorial`** for more details.

**In this section**

```{toctree}
:maxdepth: 1
local-remote
mlrun-execution-context
decorators-and-auto-logging
/runtimes/configuring-job-resources
scheduled-jobs
submitting-tasks-jobs-to-functions
workflow-overview
/runtimes/conditional-workflow
/runtimes/multiple-funcs-exithandler
/runtimes/multiple-parallel-workflow
```