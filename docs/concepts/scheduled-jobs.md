(scheduled-jobs)=
# Scheduled jobs and workflows

Oftentimes you may want to run a `job` on a regular schedule. For example, fetching from a datasource every morning, compiling an analytics report every month, or detecting model drift every hour.

> Schedules have a minimum interval that will be allowed between two scheduled jobs. By default, a job is not allowed to be scheduled twice in a 10-minute period 
> Currently, schedules like */13 * * * * (every 13th minute), in which the job would trigger at the 52nd minute and then again at the start of the next hour (minute 0) (with only 8 minutes between runs) are not allowed. 
> See mlrun.mlconf.httpdb.scheduling for service schedules configuration. 

## Creating a job and scheduling it

MLRun makes it very simple to add a schedule to a given `job`. To showcase this, the following job runs the code below, which resides in a file titled `schedule.py`:

```python
def hello(context):
    print("You just ran a scheduled job!")
```

To create the job, use the `set_function` syntax and specify the `kind` like below:

```python
import mlrun

project = mlrun.get_or_create_project("schedule")
job = project.set_function(
    name="my-scheduled-job",  # Name of the job (displayed in console and UI)
    filename="schedule.py",  # Python file or Jupyter notebook to run
    kind="job",  # Run as a job
    image="mlrun/mlrun",  # Use this Docker image
    handler="hello",  # Execute the function hello() within code.py
)
```

**Running the job using a schedule**

To add a schedule, run the job and specify the `schedule` parameter using Cron syntax like so:

```python
job.run(schedule="0 * * * *")
```

This runs the job every hour. An excellent resource for generating Cron schedules is [Crontab.guru](https://crontab.guru/).

## Scheduling a workflow

```{admonition} Note
Tech Preview
```

After loading the project (`load_project`), run the project with the scheduled workflow:

```
project.run("main", schedule='0 * * * *')
```

Remote/Scheduled workflows can be performed by a project with a remote source or one that is contained on the image. 
Remote source will be pulled each time the workflow is run, while the local source will be loaded from the image.  
To use a remote source you can either put your code in Git or archive it and then set a source to it (e.g. git://github.com/mlrun/something.git, http://some/url/file.zip, s3://some/url/file.tar.gz etc.). By default, the defined project source will be used.
* To set project source use the `project.set_source` method.
* To set workflow use the `project.set_workflow` method.  

To use a different remote source, specify the source URL when running the workflow with `project.run(source=<source-URL>)` method.  
You can also use a context path to load the project from a local directory contained in the image used for execution:
* To set project source use the `project.set_source` method (make sure `pull_at_runtime` is set to `False`).
* To build the image with the project yaml and code use `project.build_image` method. Optionally specify a `target_dir` for the project content.
* Create the workflow e.g. `project.set_workflow(name="my-workflow", workflow_path="./src/workflow.py")`.
* The default workflow image is `project.spec.default_image` which was enriched to and built with `project.build_image` unless specified otherwise.
* Run the workflow with the context path e.g. `project.run("my-workflow", source="./", engine="remote")`. The `source` can be absolute or relative path with `"."` or `"./"`.

Example for a remote GitHub project - https://github.com/mlrun/project-demo

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

You can delete a scheduled workflow in the MLRun UI. To update a scheduled workflow, re-define the schedule in the workflow, for example:

```
project.run("main", schedule='0 * * * *')
```