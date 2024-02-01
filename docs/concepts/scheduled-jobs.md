(scheduled-jobs)=
# Scheduled jobs and workflows

Oftentimes you may want to run a `job` on a regular schedule. For example, fetching from a datasource every morning, compiling an analytics report every month, or detecting model drift every hour.

## Creating a job and scheduling it

MLRun makes it very simple to add a schedule to a given `job`. To showcase this, the following job runs the code below, which resides in a file titled `schedule.py`:

```python
def hello(context):
    print("You just ran a scheduled job!")
```

To create the job, use the `code_to_function` syntax and specify the `kind` like below:

```python
import mlrun

job = mlrun.code_to_function(
    name="my-scheduled-job",      # Name of the job (displayed in console and UI)
    filename="schedule.py",       # Python file or Jupyter notebook to run
    kind="job",                   # Run as a job
    image="mlrun/mlrun",          # Use this Docker image
    handler="hello"               # Execute the function hello() within code.py
)
```

**Running the job using a schedule**

To add a schedule, run the job and specify the `schedule` parameter using Cron syntax like so:

```python
job.run(schedule="0 * * * *")
```

This runs the job every hour. An excellent resource for generating Cron schedules is [Crontab.guru](https://crontab.guru/).

## Scheduling a workflow

After loading the project (`load_project`), run the project with the scheduled workflow:

```
project.run("main", schedule='0 * * * *')
```

```{admonition} Note
1. Remote workflows can be performed by a project with a **remote** source (git://github.com/mlrun/something.git, http://some/url/file.zip or http://some/url/file.tar.gz). You can either put your code in Git or archive it and then set a source to it.
    * To set project source use the `project.set_source` method.
    * To set workflow use the `project.set_workflow` method.
2. You can also use a context path to load the project from a local directory.
    * To set project source use the `project.set_source` method (make sure pull_at_runtime is set to False).
    * To build the image with the project yaml and code use `project.build_image` method (you can specify a target_dir for the project content).
    * Specify the image in the workflow e.g. `project.set_workflow(name="my-workflow", workflow_path="./src/workflow.py", image=project.spec.default_image)`.
    * Run the workflow with the context path e.g. `project.run("my-workflow", source="/home/mlrun_code/", engine="remote")`.
    * "/home/mlrun_code/" is the default source loading path when building the image. See `project.build_image(target_dir)` to specify a different path.
3. Example for a remote GitHub project - https://github.com/mlrun/project-demo
```

You can delete a scheduled workflow in the MLRun UI. To update a scheduled workflow, re-define the schedule in the workflow, for example:

```
project.run("main", schedule='0 * * * *')
```