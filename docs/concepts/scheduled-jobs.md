(scheduled-jobs)=
# Scheduled Jobs

Often times you may want to repeadedly run a `Job` on a regular schedule. For example, fetching from a datasource every morning, compiling an analytics report every month, or detecting model drift every hour.

### Create a Job

MLRun makes it very simple to add a schedule to a given `Job`. To showcase this, we will add a schedule to run the following `schedule.py` file every hour:

```python
def hello(context):
    print("You just ran a scheduled job!")
```

To create the Job, use the `code_to_function` syntax and specify the `kind` like below:

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

### Add a Schedule

To add a schedule, we will run the job and specify the `schedule` parameter using Cron syntax like so:

```python
job.run(schedule="0 * * * *")
```

An execellent resource for generating Cron schedules is [Crontab.guru](https://crontab.guru/).