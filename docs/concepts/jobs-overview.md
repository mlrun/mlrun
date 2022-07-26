(jobs-overview)=
# Jobs Overview

A `Job` is simply something that you would like to run once to completion. For example, running a simple Python script can be similar to a `Job` in that the script runs once to completion and then returns. In an ML workflow, sometimes running a simple Python script is not enough and additional functionality is required. For example giving cluster resources, specifying dependencies and a Docker image, integrating with Git repo, etc. 

## Create a Job

MLRun can add all of the above features, and more, when running a `job`. To showcase this, the following job runs the code below, which resides in a file titled `code.py`:

```python
def hello(context):
    print("You just ran a job!")
```

To create the Job, use the `code_to_function` syntax and specify the `kind` like below:

```python
import mlrun

job = mlrun.code_to_function(
    name="my-job",                # Name of the job (displayed in console and UI)
    filename="code.py",           # Python file or Jupyter notebook to run
    kind="job",                   # Run as a job
    image="mlrun/mlrun",          # Use this Docker image
    handler="hello"               # Execute the function hello() within code.py
)
```

Read more about the {py:meth}`~mlrun.run.code_to_function` syntax.

## Run a Job Locally

When prototyping, it is often useful to test the `Job` locally on your laptop or Jupyter environment before running on the larger cluster. This lets you ensure the job does what you want without using cluster resources.

To do this, run the job and specify the `local=True` flag 

```python
run = job.run(local=True)
```

## Run a Job on the Cluster

Finally, you can execute your job using cluster resources. This is usually the end goal when creating a job because it gives you much more flexibility into the configuration of the job.

To do this, run the job and specify the `local=False` flag or omit the `local` flag all together:

```python
run = job.run(local=False)
```

## Configure the Job

There are many configurations you can add to the `Job`. You can read more about them here:
- [Customize Docker image and dependencies](#) **PAGE DOES NOT EXIST**
- [Add CPU, GPU, Memory resources](https://github.com/mlrun/mlrun/pull/2166/runtimes/configuring-job-resources.html)
- Use alternative runtimes including [Dask](https://github.com/mlrun/mlrun/pull/2166/runtimes/dask-overview.html), [Horovod](https://github.com/mlrun/mlrun/pull/2166/runtimes/horovod.html), and [Spark](https://github.com/mlrun/mlrun/pull/2166/runtimes/spark-operator.html)
- [Schedule a Job](https://github.com/mlrun/mlrun/pull/2166/files/scheduled-jobs.html)
- [Attach storage to a Job](https://github.com/mlrun/mlrun/pull/2166/runtimes/function-storage.html)
- [Run a Job with a Git repo](https://github.com/mlrun/mlrun/pull/2166/runtimes/code-archive.html#using-code-from-git)