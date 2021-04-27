# Functions & Runs

All the executions in MLRun are based on Serverless Functions, the functions allow specifying code and 
all the operational aspects (image, required packages, cpu/mem/gpu resources, storage, environment, etc.), 
the different function runtimes take care of automatically transforming the code and spec to fully 
managed and elastic services over Kubernetes which save significant operational overhead, 
address scalability and reduce infrastructure costs.

Function objects are all inclusive (code +  spec + API and metadata definitions) which allow placing them 
in a shared and versioned function market place, this way different members of the team can produce or 
consume functions. Each function is versioned and stored in the MLRun database with a unique hash code, 
and gets a new hash code upon changes.
There is also an open [public marketplace](https://github.com/mlrun/functions) which store many pre-developed functions for
use in your projects. 

Functions (function objects) can be created by using any of the following methods:

- **`new_function`** - creates a function "from scratch" or from another function.
- **`code_to_function`** - creates a function from local or remote source code or from a notebook.
- **`import_function`** - imports a function from a local or remote YAML function-configuration file or 
  from a function object in the MLRun database (using a DB address of the format `db://<project>/<name>[:<tag>]`)
  or from the function marketplace (e.g. `hub://describe`).

You can use the `.save` function method to save a function object in the MLRun database, or 
the `.export` method to save a YAML function-configuration to your preferred local or remote location.
Use `.deploy` to build/deploy the function (if we need to add packages or run it as an online service), 
`.run` to execute a task, `.as_step` to convert a function to a Kubeflow pipeline step.
For function-method details and examples, see the embedded documentation/help text.

Functions are stored in the project and are versioned. Therefore, you can always view previous code and go back to previous functions if needed.

MLRun supports batch functions (based on Kubernetes jobs, Spark, Dask, Horovod, etc.) or Real-time functions 
for serving, APIs, and stream processing (based on the high-performance Nuclio engine).

Batch functions accept **Tasks** (a set of parameters, inputs, secrets, etc.) and return a **Run objects** 
which hosts the status, results, data outputs, logs, etc. every execution has a unique Run ID.
Tasks can be broken to multiple child tasks (called `Iterations`), allowing to run a sequence of 
hyper-parameter or AutoML jobs. 

**MLRun Functions and Tasks**

<img src="../_static/images/mlrun-functions.png" alt="mlrun-architecture" width="800"/>

## Specify/Import Function Code

In MLRun code can be provided in several ways:
1. inline as part of the function object 
2. loaded into the function container as part of the build/deploy process 
3. loaded from git/zip/tar archive into the function at runtime 

the first option is great for small and single file functions or for using code derived from notebooks, we use mlrun 
:py:func:`~mlrun.code_to_function` method to create functions from code files or notebooks.

the build/deploy option is good for making sure we have an integrated code + dependencies container package and avoid 
the dependency or overhead of loading code at runtime. We need to make sure we add the source archive into our container or use the `function.deploy()` method which will build a container for us, we can specify the build configuration using the `function.build_config()` method. 

when doing iterative development with multiple code files and packages the 3rd option is the most efficient, we want 
to make small code changes and re-run our job without building containers etc.

the `local`, `job` and `remote-spark` runtimes support dynamic load from archive or file shares (other runtimes will 
be added later), this is enabled by setting the `spec.build.source=<archive>` and `spec.build.load_source_on_run=True` 
or by using the `function.from_source_archive()` method. in the CLI we use the `--source` flag. 

see more details and examples on [**running jobs with code from Archives or shares**](./code-archive.ipynb)

## Running Tasks Using Functions

MLRun batch Function objects support a `.run()` method for invoking a job over them, the run method 
accepts various parameters such as `name`, `handler`, `params`, `inputs`, `schedule`, etc. 
Alternatively we can pass a `Task` object which holds all of our parameters. 

Functions may host multiple methods (handlers), we can set the default handler per function, 
 we need to specify which handle we intend to call in the run command. 
 
The `run` command returns a run object which allowed us to track our job and its results, when we 
pass the parameter `watch=True` (default) the `run()` command will block until our job completes.

Run object has the following methods:
- `uid` &mdash; returns the unique ID.
- `state` &mdash; returns the last known state.
- `show` &mdash; shows the latest job state and data in a visual widget (with hyperlinks and hints).
- `outputs` &mdash; returns a dictionary of the run results and artifact paths.
- `logs` &mdash; returns the latest logs.
    Use `Watch=False` to disable the interactive mode in running jobs.
- `artifact` &mdash; returns full artifact details for the provided key.
- `output` &mdash; returns a specific result or an artifact path for the provided key.
- `to_dict`, `to_yaml`, `to_json` &mdash; converts the run object to a dictionary, YAML, or JSON format (respectively).

## MLRun Execution Context

In the function code signature we can add the `context` attribute (first), this provides us access to the 
job metadata, parameters, inputs, secrets, and API for logging and monitoring our results. 
Alternatively if we dont run inside a function handler (e.g. in Python main or Notebook) we can obtain the `context` 
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

## Supported Runtimes

MLRun supports the following Runtimes
* **Handler** - execute python handelr (used in notebooks or debug)
* **Local** - execute a Python or shell program 
* **Job** - run the code in a Kubernetes Job
* **Dask** - run the code as a Dask Distributed job (over Kubernetes)
* **MPIJob** - run distributed jobs and Horovod over the MPI job operator, used mainly for deep learning jobs 
* **Spark** - run the job as a Spark job (using Spark Kubernetes Operator)
* **Nuclio** - real-time serverless functions over Nuclio
* **Serving** - higher level real-time Graph (DAG) over one or more Nuclio functions

Functions are associated with a specific runtime, and every runtime may add specific attributes 
(e.g. Jars for Spark, Triggers for Nuclio, Auto-scaling for Dask, etc.), check the runtime specific 
documentation for details.

## Kubernetes Based Functions 

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

## Elastic Serverless Runtimes

Many of the runtimes support horizontal scaling, you can specify the number of `replicas` or the 
min - max value range (for auto scaling in Dask or Nuclio). When scaling functions we use some high speed
messaging protocol and shared storage (volumes, objects, databases, or streams). MLRun runtimes
handle the orchestration and monitoring of the distribured task.

<img src="../_static/images/runtime-scaling.png" alt="runtime-scaling" width="400"/>

