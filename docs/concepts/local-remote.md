(local-remote)=
# Local vs. remote workflows

To run multiple functions, one after the other  (`jobs`), you use a pipeline. There are three types of pipeline engines:
- [Remote on KFP](#remote-kfp)
- [KFP](#kfp)
- Local &mdash; Used to run local pipeline with local functions, mainly for testing. Use (set `local=True in function.run()`).

<img src="../_static/images/pipelines-flow.png" width="800" >

All three types are configured when running the workflow, see {py:class}`mlrun.projects.MlrunProject.run`.
 
## Remote-KFP 

Remote workflows are run on the remote server with KFP. Remote workflows are used for [scheduled workflows](./scheduled-jobs.md#scheduling-a-workflow).
The remote workflow supports [sending notifications](./notifications.md#remote-pipeline-notifications) when runs are complete.
 

Remote workflows must be based on the image `mlrun/mlrun-kfp`. See {ref}`images-usage`.  
Remote workflows can be run by a project with a remote source or one that is contained in the image. See [source files](../runtimes/create-and-use-functions.ipynb#multiple-source-files).  
To set the workflow, use the method {py:class}`mlrun.projects.MlrunProject.set_workflow` for example: `project.set_workflow(name="my-workflow", workflow_path="./src/workflow.py")`.  

A remote workflow creates a pod for each job, and a pod for the KFP. The job pod is named `workflow-runner-<workflow-name>`. You can modify the pod image and the pod node selector with:
- `project.set_workflow(name="main",workflow_path="workflow.py,image="<runner-image>")` &mdash; changing the runner image
- `project.run("main",engine="remote",workflow_runner_node_selector={"key":"value"})` &mdash; changing the node selector image

See an example of a remote GitHub project in https://github.com/mlrun/project-demo.
```{admonition} Note
From MLRun v1.7.1: when running a remote/scheduled workflow, the remote workflow pulls/extracts the remote source content to the running pod but loads the project configuration from the MLRun DB and not from the project.yaml file in the remote source.

The remote files are primarily retrieved for:
- The [project_setup](../projects/project-setup.md) that may affect the project configuration (if it exists).
- Syncing function files.
This behavior may be unexpected for users who rely on project.yaml in the remote source (for the project configuration).
Be sure to update MLRun DB with the latest project configuration to ensure consistent configuration management (use `project.save()`).<br>
Project configuration in this context could be, for example, `project.node_selector` or `project.artifact_path`, and not function configurations like: function resources or function node selector.
```
```
import mlrun
project_name = "remote-workflow-example"
source_url = "git://github.com/mlrun/project-demo.git"
source_code_target_dir = "./project" # Optional, relative to "/home/mlrun_code". A different absolute path can be specified.

# Create a new project
project = mlrun.load_project(context=f"./{project_name}", url=source_url, name=project_name)

# Set the project source and workflow
project.set_source(source_url)
project.set_workflow(name="main", workflow_path="kflow.py")

# Build the image, load the source to the target dir and save the project
project.build_image(target_dir=source_code_target_dir)
project.save()

# Run the workflow, load the project from the target dir on the image
project.run("main", source="./", engine="remote", dirty=True)
```

See also [Local and KFP engine pipeline notifications](../concepts/notifications.md#local-and-kfp-engine-pipeline-notifications) and [Setting notifications on scheduled run](../concepts/notifications.md#setting-notifications-on-scheduled-runs).

## KFP

The KFP workflow spec file is created in MLRun, and is compiled and run in the client side. 
For example:
```
project.run("main", engine='remote')
```
