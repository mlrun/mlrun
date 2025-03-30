(local-remote)=
# Local vs. remote workflows

To run a pipeline, you need to run a workflow. There are two types of workflows:
- [Remote on KFP](#remote-kfp)
- [Remote](#remote-workflows)
- [Local](#local-workflows)

<img src="../_static/images/pipelines-flow.png" width="800" >

## Remote-KFP 

The pipeline runs on the remote server using KFP. You can use either a remore source, or a local source.
The your source is local, then you need to install KFP locally. In this case, you don't have to commit the code.

Remote workflows must be based on the image 'mlrun/mlrun-kfp`. (See {ref}`images-usage`.) 

The remote workflow supports sending notifications, and???

Useful: for schedules

There are several ways to run a remote-KFP workflow::
- With Git: For each run of the schedule, it clones the code, compiles the pipeline, and then runs it.
- Build the image with the source code of the pipeline, compile the pipeline, and then run it.

There is one pod for each function, and also a batch pod for each function.

## Remote workflows

The pipeline runs in KFP: the spec file is sent to KFP to compile the pipeline, and run the workflow.

The code of the pipeline is compiled in your environment, then sent to the MLRun
API, which sends it to the KFP.

The pipeline compiles the code on the pod, and then runs the workflow. 

There is one pod for each function, and also a batch pod for each function.

## Local workflows

Local workflows are useful when you want to debug the flow of the pipeline code itself. 

The  code is compiled locally and the functions run on the remote KFP. KFP must be installed locally. 

Running workflows locally uses a completely different environment, for example, a different python version and different packages based on the local environment.

This option is configured by setting `local=True`.

There are pods for each function.