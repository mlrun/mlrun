
## Creating and Using Functions

**Functions** (function objects) can be created by using any of the following methods:

- **{py:func}`~mlrun.run.new_function`** - creates a function for local run or from container, from code repository/archive, from function spec.
- **{py:func}`~mlrun.run.code_to_function`** - creates a function from local or remote source code (single file) or from a notebook (code file will be embedded in the function object).
- **{py:func}`~mlrun.run.import_function`** - imports a function from a local or remote YAML function-configuration file or 
  from a function object in the MLRun database (using a DB address of the format `db://<project>/<name>[:<tag>]`)
  or from the function marketplace (e.g. `hub://describe`).

You can use the {py:meth}`~mlrun.runtimes.BaseRuntime.save` function method to save a function object in the MLRun database, or 
the {py:meth}`~mlrun.runtimes.BaseRuntime.export` method to save a YAML function-configuration to your preferred local or remote location.
use {py:meth}`~mlrun.runtimes.BaseRuntime.run` to execute a task, or {py:meth}`~mlrun.runtimes.BaseRuntime.as_step` to convert a function to a Kubeflow pipeline step.
Use `.deploy()` to build/deploy the function (deploy for batch functions will build the image and add required packages, 
for online/real-time runtimes like `nuclio` and `serving` it will also deploy it as an online service)
For method details click the hyperlinks or check the embedded documentation/help text.

Functions are stored in the project and are versioned. Therefore, you can always view previous code and go back to previous functions if needed.

* [**Specifying Function Code**](#specifying-function-code)
* [**Submitting Tasks/Jobs To Function**](#)
* [**MLRun Execution Context**](#mlrun-execution-context)
* [**Function Runtimes**](#function-runtimes)

### Specifying Function Code

In MLRun code can be provided in several ways:
1. inline as part of the function object 
2. loaded into the function container as part of the build/deploy process 
3. loaded from git/zip/tar archive into the function at runtime 

the first option is great for small and single file functions or for using code derived from notebooks, we use mlrun 
{py:func}`~mlrun.code_to_function` method to create functions from code files or notebooks.
For more on how to create functions from notebook code, see [converting notebook code to a function](./mlrun_code_annotations.ipynb).

    # create a function from py or notebook (ipynb) file, specify the default function handler
    my_func = mlrun.code_to_function(name='prep_data', filename="./prep_data.py", 
                                     kind='job', image='mlrun/mlrun', handler='my_func')

    # add shared storage volume to it for reading/writing data.
    # only needed if specific storage configuration is needed, that is not supplied by auto-mount
    # my_func.apply(mount_v3io())

    # run the function
    run_results = my_func.run(params={"label_column": "label"}, inputs={'data': data_url})

the build/deploy option is good for making sure we have a container package with integrated code + dependencies and avoid 
the dependency or overhead of loading code at runtime. We need to make sure we add the source archive into our container 
or use the {py:meth}`~mlrun.runtimes.KubejobRuntime.deploy()` method which will build a container for us, we can specify 
the build configuration using the {py:meth}`~mlrun.runtimes.KubejobRuntime.build_config` method. 

    # create a new job function from base image and archive + custom build commands
    fn = mlrun.new_function('archive', kind='job', command='./myfunc.py')
    fn.build_config(base_image='mlrun/mlrun', source='git://github.com/org/repo.git#master',
                    commands=[pip install pandas])
    # deploy (build the container with the extra build commands/packages)
    fn.deploy()
    
    # run the function (specify the function handler to execute)
    run_results = fn.run(handler='my_func', params={"x": 100})

The `command='./myfunc.py'` specifies the command we execute in the function container/workdir, by default we call python 
with the specified command, you can specify `mode="pass"` to execute the command as is (e.g. for binary code), you can 
template (`{..}`) in the command to pass the task parameters as arguments for the execution command (e.g. `mycode.py --x {xparam}` will 
substitute the `{xparam}` with the value of the `xparam` parameter) 

when doing iterative development with multiple code files and packages the 3rd option is the most efficient, we want 
to make small code changes and re-run our job without building containers etc.

the `local`, `job`, `mpijob` and `remote-spark` runtimes support dynamic load from archive or file shares (other runtimes will 
be added later), this is enabled by setting the `spec.build.source=<archive>` and `spec.build.load_source_on_run=True` 
or simply by setting the `source` attribute in `new_function`). in the CLI we use the `--source` flag. 

    fn = mlrun.new_function('archive', kind='job', image='mlrun/mlrun', command='./myfunc.py', 
                            source='git://github.com/mlrun/ci-demo.git#master')
    run_results = fn.run(handler='my_func', params={"x": 100})

see more details and examples on [**running jobs with code from Archives or shares**](./code-archive.ipynb)

For executing non-python code, set `mode="pass"` (passthrough) and specify the full execution `command`, e.g.:

    new_function(... command="bash main.sh --myarg xx", mode="pass")  

### Submitting Tasks/Jobs To Functions

MLRun batch Function objects support a {py:meth}`~mlrun.runtimes.BaseRuntime.run` method for invoking a job over them, the run method 
accept various parameters such as `name`, `handler`, `params`, `inputs`, `schedule`, etc. 
Alternatively we can pass a **`Task`** object (see: {py:func}`~mlrun.model.new_task`) which holds all of our parameters plus advanced options. 

```{admonition} Run/simulate functions locally
Functions can also run and be debugged locally by using the `local` runtime or by setting the `local=True` parameter in the {py:meth}`~mlrun.runtimes.BaseRuntime.run` 
method (for batch functions)
```

Functions may host multiple methods (handlers), we can set the default handler per function, 
 we need to specify which handle we intend to call in the run command. 
 
Users can pass data objects to functions using the `inputs` dictionary argument with the data input key (as specified in the function handler)
and the MLRun data url, the data will be passed into the function as a {py:class}`~mlrun.datastore.DataItem` object which handles data movement, 
tracking and security in an optimal way (read more about data objects in: [Data Stores & Data Items](../store/datastore.md))

    run_results = fn.run(params={"label_column": "label"}, inputs={'data': data_url})

MLRun also support iterative jobs which can run and track multiple child jobs (for hyper-parameter tasks, AutoML, etc.), 
see [Hyper-Param and Iterative jobs](../hyper-params.ipynb) for details and examples.
 
The `run()` command returns a run object which allowed us to track our job and its results, when we 
pass the parameter `watch=True` (default) the {py:meth}`~mlrun.runtimes.BaseRuntime.run` command will block until our job completes.

Run object has the following methods/properties:
- `uid()` &mdash; returns the unique ID.
- `state()` &mdash; returns the last known state.
- `show()` &mdash; shows the latest job state and data in a visual widget (with hyperlinks and hints).
- `outputs` &mdash; returns a dictionary of the run results and artifact paths.
- `logs(watch=True)` &mdash; returns the latest logs.
    Use `Watch=False` to disable the interactive mode in running jobs.
- `artifact(key)` &mdash; returns artifact for the provided key (as {py:class}`~mlrun.datastore.DataItem` object).
- `output(key)` &mdash; returns a specific result or an artifact path for the provided key.
- `wait_for_completion()` &mdash; wait for async run to complete
- `refresh()` &mdash; refresh run state from the db/service
- `to_dict()`, `to_yaml()`, `to_json()` &mdash; converts the run object to a dictionary, YAML, or JSON format (respectively).


<br>You can view the job details, logs and artifacts in the user interface:

<br><img src="../_static/images/project-jobs-train-artifacts-test_set.png" alt="project-jobs-train-artifacts-test_set" width="800"/>


### MLRun Execution Context

In the function code signature we can add the `context` attribute (first), this provides us access to the 
job metadata, parameters, inputs, secrets, and API for logging and monitoring our results. 
Alternatively if we don't run inside a function handler (e.g. in Python main or Notebook) we can obtain the `context` 
object from the environment using the {py:func}`~mlrun.run.get_or_create_ctx` function.

example function and usage of the context object:
 
```python
from mlrun.artifacts import ChartArtifact
import pandas as pd

def my_job(context, p1=1, p2="x"):
    # load MLRUN runtime context (will be set by the runtime framework)

    # get parameters from the runtime context (or use defaults)

    # access input metadata, values, files, and secrets (passwords)
    print(f"Run: {context.name} (uid={context.uid})")
    print(f"Params: p1={p1}, p2={p2}")
    print("accesskey = {}".format(context.get_secret("ACCESS_KEY")))
    print("file\n{}\n".format(context.get_input("infile.txt", "infile.txt").get()))

    # Run some useful code e.g. ML training, data prep, etc.

    # log scalar result values (job result metrics)
    context.log_result("accuracy", p1 * 2)
    context.log_result("loss", p1 * 3)
    context.set_label("framework", "sklearn")

    # log various types of artifacts (file, web page, table), will be versioned and visible in the UI
    context.log_artifact(
        "model",
        body=b"abc is 123",
        local_path="model.txt",
        labels={"framework": "xgboost"},
    )
    context.log_artifact(
        "html_result", body=b"<b> Some HTML <b>", local_path="result.html"
    )

    # create a chart output (will show in the pipelines UI)
    chart = ChartArtifact("chart")
    chart.labels = {"type": "roc"}
    chart.header = ["Epoch", "Accuracy", "Loss"]
    for i in range(1, 8):
        chart.add_row([i, i / 20 + 0.75, 0.30 - i / 20])
    context.log_artifact(chart)

    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
        "age": [42, 52, 36, 24, 73],
        "testScore": [25, 94, 57, 62, 70],
    }
    df = pd.DataFrame(raw_data, columns=["first_name", "last_name", "age", "testScore"])
    context.log_dataset("mydf", df=df, stats=True)
```

example, creating the context objects from the environment:

```python
if __name__ == "__main__":
    context = mlrun.get_or_create_ctx('train')
    p1 = context.get_param('p1', 1)
    p2 = context.get_param('p2', 'a-string')
    # do something
    context.log_result("accuracy", p1 * 2)
    # commit the tracking results to the DB (and mark as completed)
    context.commit(completed=True)
```

Note that MLRun context is also a python context and can be used in a `with` statement (eliminating the need for `commit`)

```python
if __name__ == "__main__":
    with mlrun.get_or_create_ctx('train') as context:
        p1 = context.get_param('p1', 1)
        p2 = context.get_param('p2', 'a-string')
        # do something
        context.log_result("accuracy", p1 * 2)
```

(Function_runtimes)=
### Function Runtimes

When users create MLRun functions they need to specify one of the following function runtime kinds (e.g. `kind='job'`):
* **handler** - execute python handler (used automatically in notebooks or for debug)
* **local** - execute a Python or shell program 
* **job** - run the code in a Kubernetes Pod
* **dask** - run the code as a Dask Distributed job (over Kubernetes)
* **mpijob** - run distributed jobs and Horovod over the MPI job operator, used mainly for deep learning jobs 
* **spark** - run the job as a Spark job (using Spark Kubernetes Operator)
* **remote-spark** - run the job on a remote Spark service/cluster (e.g. Iguazio Spark service)
* **nuclio** - real-time serverless functions over Nuclio
* **serving** - higher level real-time Graph (DAG) over one or more Nuclio functions

Functions are associated with a specific runtime, and every runtime may add specific attributes 
(e.g. Jars for Spark, Triggers for Nuclio, Auto-scaling for Dask, etc.), check the runtime specific 
documentation links below for details.

**Common attributes for Kubernetes based functions** 

All the Kubernetes based runtimes (Job, Dask, Spark, Nuclio, MPIJob, Serving) support a common 
set of spec attributes and methods for setting the PODs:

function.spec attributes (similar to k8s pod spec attributes):
* volumes
* volume_mounts
* env
* resources
* replicas
* image_pull_policy
* service_account
* image_pull_secret

common function methods:
* set_env(name, value)
* set_envs(env_vars)
* gpus(gpus, gpu_type)
* with_limits(mem, cpu, gpus, gpu_type)
* with_requests(mem, cpu)
* set_env_from_secret(name, secret, secret_key)
