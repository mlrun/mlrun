(projects)=
# Projects and automation

MLRun **Project** is a container for all your work on a particular ML application. Projects host [functions](../runtimes/functions.md), [workflows](../concepts/workflow-overview.md), [artifacts (datasets, models, etc.)](../store/artifacts.md), [features (sets, vectors)](../feature-store/feature-store.md), 
and configuration (parameters, [secrets](../secrets.md), source, etc.). Projects have owners and members with role-based access control.

<p align="center"><img src="../_static/images/project.png" alt="mlrun-project" width="600"/></p><br>

Projects are stored in a GIT or archive and map to IDE projects (in PyCharm, VSCode, etc.), which enables versioning, collaboration, and [CI/CD](../projects/ci-integration.md). 
Projects simplify how you process data, [submit jobs](../concepts/submitting-tasks-jobs-to-functions.md), run [multi-stage workflows](../concepts/workflow-overview.md), and deploy [real-time pipelines](../serving/serving-graph.md) in continuous development or production environments.

<p align="center"><img src="../_static/images/project-lifecycle.png" alt="project-lifecycle" width="700"/></p><br>

**In this section**

```{toctree}
:maxdepth: 1

create-project
git-best-practices
load-project
run-build-deploy
build-run-workflows-pipelines
../secrets
project-setup
```

**See also:**
- {ref}`ci-cd-automate`
- {ref}`ci-integration`