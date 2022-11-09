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

Before saving the project (`project.save`), add the schedule to the workflow. This example runs the workflow every hour:

```
main_workflow = project.spec.workflows[0]
main_workflow["schedule"] = '0 * * * *'
project.spec.set_workflow("main", main_workflow)
```

After loading the project (`load_project`), run the project with the scheduled workflow:

```
project.run("main", schedule=True)
```
