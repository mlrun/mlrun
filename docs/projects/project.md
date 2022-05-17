(Projects)=
# Projects

A project is a container for all your work on a particular activity/application. It is the basic starting point for your work.
Project definitions include all of the associated code, [functions](../runtimes/functions), [Submitting tasks/jobs to functions](../concepts/submitting-tasks-jobs-to-functions), [artifacts](../store/artifacts), lists of parameters, and secrets.
You can create project definitions using the SDK or a yaml file and store those in the MLRun DB, a file, or an archive.  Project 
jobs/workflows refer to all project resources by name, establishing a separation between configuration and code.

Projects can be mapped to `git` repositories or IDE project (in PyCharm, VSCode, etc.), which enables versioning, collaboration, and CI/CD. 
Project access can be restricted to a set of users and roles.

Projects can be loaded/cloned using a single command. Once the project is loaded you can execute the functions or workflows locally, on a cluster, or inside a CI/CD framework.

**In this section**
```{toctree}
:maxdepth: 2

create-load-import-project
use-project
workflows
../secrets
```
