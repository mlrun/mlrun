(projects)=
# Projects and automation

A project is a container for all your work on a particular activity/application. It is the basic starting point for your work.
[Project definitions](create-load-import-project.html) include all of the associated code, [functions](../runtimes/functions.html), [submitting tasks/jobs to functions](../concepts/submitting-tasks-jobs-to-functions.html), [artifacts](../store/artifacts.html), 
lists of parameters, and [secrets](../secrets.html).
You can create project definitions using the SDK or a yaml file and store those in the MLRun DB, a file, or an archive.  Project 
jobs/workflows refer to all project resources by name, establishing a separation between configuration and code.

Projects can be mapped to `git` repositories or IDE project (in PyCharm, VSCode, etc.), which enables versioning, collaboration, and [CI/CD](../projects/ci-integration.html). 
Project access can be restricted to a set of users and roles.

Projects can be loaded/cloned using a single command. Once the project is loaded you can execute the functions or workflows locally, 
on a cluster, or inside a CI/CD framework.

**In this section**

```{toctree}
:maxdepth: 1

create-load-import-project
use-git-manage-projects
use-project
run-build-deploy
build-run-workflows-pipelines
ci-integration
../secrets
```