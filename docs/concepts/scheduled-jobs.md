(scheduled-jobs)=
# Scheduled jobs and workflows

Oftentimes you may want to run a `job` on a regular schedule, for example, fetching from a datasource every morning, compiling an analytics report every month, or detecting model drift every hour.

- Schedules have a minimum interval that will be allowed between two scheduled jobs. By default, a job is not allowed to be scheduled twice in a 10-minute period.
- Currently, schedules like */13 * * * * (every 13th minute), in which the job would trigger at the 52nd minute and then again at the start of the next hour (minute 0) (with only 8 minutes between runs) are not allowed. 
- See scheduling in the [`mlrun config`](https://github.com/mlrun/mlrun/blob/development/mlrun/config.py#L52) for more details. 

**In this section**

- [Creating a job and scheduling it](#creating-a-job-and-scheduling-it)
- [Scheduling a workflow](#scheduling-a-workflow)

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
Scheduled jobs are essentially [remote-KFP workflows](./local-remote.md#remote-kfp) with a schedule. 
After loading the project (`load_project`), run the project with the scheduled workflow:

```
project.run("main", schedule='0 * * * *')
```

You can delete a scheduled workflow in the MLRun UI. To update a scheduled workflow, re-define the schedule in the workflow, for example:

```
project.run("main", schedule='0 * * * *')
```