(jobs)=
# Jobs

A `job` is simply something that you would like to run once to completion. For example, running a simple Python script can be similar to a 
`job` in that the script runs once to completion and then returns. In an ML workflow, sometimes running a simple Python script is not enough 
and additional functionality is required. For example giving cluster resources, specifying dependencies and a Docker image, integrating with Git repo, etc. 

**In this section**

```{toctree}
:maxdepth: 1

jobs-overview
mlrun-execution-context
submitting-tasks-jobs-to-functions
../runtimes/mlrun_jobs
using-tasks-secrets
scheduled-jobs
../runtimes/distributed
node-affinity
../runtimes/configuring-job-resources
```