(local-remote)=
# Types of workflows
A workflow is a Python file that defines and triggers a series of MLRun jobs. There are three types of workflows, depending on where the code is compiled and executed.
- [Remote on KFP](#remote-kfp)
- [KFP](#kfp)
- [Local](#local)

<img src="../_static/images/pipelines-flow.png" width="800" >

All three types are configured by the `engine` flag, when running the workflow. Ssee {py:meth}`mlrun.projects.MlrunProject.run`.

- **Compiling** a workflow is converting a workflow file into runnable MLRun steps. This refers to executing the Python workflow file itself.
- **Running** a workflow is executing the MLRun jobs that are defined as steps within the workflow.

**In this section**
- [Remote on KFP](#remote-kfp)
- [KFP](#kfp)
- [Local](#local)
 
## Remote-KFP 

```{admonition} Notes
- Starting from MLRun 1.9.1, the project default image no longer affects the workflow runner image.
- Remote workflows support both Python 3.9 and 3.11, but the workflow runner itself runs on Python 3.9.
- Starting from MLRun 1.8.0, the default workflow runner image is `mlrun/mlrun-kfp`. This image includes MLRun and KFP, but does not include custom packages. See {ref}`images-usage`.
```

The default Remote-KFP workflows are run on the workflow runner pod, which runs and loads your workflow on a pod named `workflow-runner-<workflow-name>` using the workflow file that is stored in a remote source (e.g. Git, tar.gz or zip). This pod is responsible for loading the files from the remote source and running the KFP by using the files from the remote source. Each step runs as a separate pod.  
If your workflow file imports custom packages, they must be included in the workflow runner image. Use one of the {py:meth}`~mlrun.projects.MlrunProject.build_image` parameters: `requirements` or `requirements_file` to add the packages.

You can modify the:
- workflow runner image: `project.set_workflow(name="main",workflow_path="workflow.py",image="<runner-image>")`
- runner node selector : `project.run("main",engine="remote",workflow_runner_node_selector={"key":"value"})`
- runner source: `project.run(source=<source-URL>)`

In some cases you might not want to load the files from the remote source, but instead use the files within the running image (see details in [build image](../projects/run-build-deploy.md#build_image)). In this case, you need to build an image that contains the workflow file and then change the workflow runner source to point to the project local files in the running image. See the example below.

Remote workflows are used for [scheduled workflows](./scheduled-jobs.md#scheduling-a-workflow). Only workflows that use the remote engine can be scheduled. 

The remote workflow supports [sending notifications](./notifications.md#remote-pipeline-notifications) when runs are complete.

See an example of a remote GitHub project in https://github.com/mlrun/project-demo.
```{admonition} Note
From MLRun v1.7.1: when running a remote/scheduled workflow, the remote workflow pulls/extracts the remote source content to the running pod but loads the project configuration from the MLRun DB and not from the `project.yaml` file in the remote source. The remote files are primarily retrieved for:
- The [project_setup](../projects/project-setup.md) that may affect the project configuration (if it exists).
- Syncing function files.
This behavior may be unexpected for users who rely on `project.yaml` in the remote source (for the project configuration).
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

# Set the project source
project.set_source(source_url)

# Build the image based on mlrun-kfp, load the source to the target dir
result = project.build_image(base_image="mlrun/mlrun-kfp" ,target_dir=source_code_target_dir, set_as_default=False)

# Set the workflow and save the project
project.set_workflow(name="main", workflow_path="kflow.py", image=result.outputs["image"])
project.save()

# Run the workflow, load the project from the target dir on the image
project.run("main", source="./", engine="remote", dirty=True)
```

See also 
- [Local and KFP engine pipeline notifications](../concepts/notifications.md#local-and-kfp-engine-pipeline-notifications)

## KFP

The KFP workflow spec file is created in MLRun, and is compiled and run in the client side, using the files from your local file system.
For example:
```
project.run("main", engine='kfp')
```

## Local
Local workflows are used to simulate a pipeline run without using KFP. Both compilation and execution happen locally on your machine: the workflows run like regular Python scripts in your IDE or Jupyter Notebook. Local workflows are used mainly for testing. 

Local workflows requires Python 3.9.

Use `local=True` in `function.run()` to run the functions locally or `project.run(local=True)` to apply for all functions.
Starting from MLRun 1.8+, you must install the KFP package locally: `pip install mlrun[kfp18]`.
If your workflow uses additional packages, they must also be installed locally.

Kubeflow specific features such as `set_retry()` are not supported. 
Docker images and project default images do not affect this type, since everything runs locally.