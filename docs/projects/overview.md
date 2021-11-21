# Projects, Automation & CI/CD

A Project is a container for all your work on a particular activity/application. All the associated code, functions, 
jobs/workflows and artifacts are organized within the projects. Projects can be mapped to `GIT` repositories or IDE project 
(in PyCharm, VSCode, ..) which enable versioning, collaboration, and CI/CD. Project access can be restricted to a set of users and roles. 

Users can create project definitions using the SDK or a Yaml file and store those in MLRun DB, file, or archive.
Project definitions include lists of parameters, functions, workflows (pipelines), artifacts, and secrets. 
Project jobs/workflows refer to any project resource by name, allowing separation between configuration and code.

Projects can be loaded/cloned using a single command, once the project is loaded you can execute the functions or 
workflows locally, on a cluster, or inside a CI/CD framework. 

```{toctree}
:maxdepth: 1

project
workflows
ci-integration
```
