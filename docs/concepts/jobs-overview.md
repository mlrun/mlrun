(jobs-overview)=
# Jobs overview

**In this section**
- [Create a job](#create-a-job)
- [Run a job locally](#run-a-job-locally)
- [Run a job on the cluster](#run-a-job-on-the-cluster)
- [Configure the job](#configure-the-job)

## Create a job

MLRun can add all of the above features, and more, when running a `job`. To showcase this, the following job runs the code below, which resides in a file titled `code.py`:

```python
def hello(context):
    print("You just ran a job!")
```

To create the job, use the `code_to_function` syntax and specify the `kind` like below:

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

## Run a job locally

When prototyping, it is often useful to test the `job` locally on your laptop or Jupyter environment before running on the larger cluster. 
In this way you can ensure that that the job does what you want without using cluster resources.

To do this, run the job and specify the `local=True` flag: 

```python
run = job.run(local=True)
```

## Run a job on the cluster

Finally, you can execute your job using cluster resources. This is usually the end goal when creating a job because it gives you much more 
flexibility into the configuration of the job.

To do this, run the job and specify the `local=False` flag or omit the `local` flag altogether:

```python
run = job.run(local=False)
```

## Configure the job

There are many configurations you can add to the `Job`. Read more about them here:
<!-- [Customize Docker image and dependencies](#) **PAGE DOES NOT EXIST** -->
- {ref}`configuring-job-resources`
- Use alternative runtimes including {ref}`Dask <dask-overview>`, {ref}`Horovod <horovod>`, {ref}`Spark <spark-operator>`
- {ref}`scheduled-jobs`
- {ref}`Attach storage to a job <Function_storage_auto_mount>`
- [Run a Job with a Git repo](../runtimes/code-archive.html#using-code-from-git)