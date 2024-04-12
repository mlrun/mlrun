(configuring-job-resources)=
# Configuring runs and functions

MLRun orchestrates serverless functions over Kubernetes. You can specify the resource requirements (CPU, memory, GPUs),
preferences, and pod priorities in the logical function object. You can also configure how MLRun prevents stuck pods.
All of these are used during the function deployment.

Configuring runs and functions is relevant for all supported cloud platforms.

**In this section**
- [Environment variables](#environment-variables)
- [Replicas](#replicas)
- [CPU, GPU, and memory limits for user jobs](#cpu-gpu-and-memory-limits-for-user-jobs)
- [Number of workers and GPUs](#number-of-workers-and-gpus)
- [Volumes](#volumes)
- [Preemption mode: Spot vs. On-demand nodes](#preemption-mode-spot-vs-on-demand-nodes)
- [Pod priority for user jobs](#pod-priority-for-user-jobs)
- [Node selection](#node-selection)
- [Scaling and auto-scaling](#scaling-and-auto-scaling)
- [Mounting persistent storage](#mounting-persistent-storage)
- [Preventing stuck pods](#preventing-stuck-pods)

## Environment variables

Environment variables can be added individually, from a Python dictionary, or a file:

```python
# Single variable
fn.set_env(name="MY_ENV", value="MY_VAL")

# Multiple variables
fn.set_envs(env_vars={"MY_ENV" : "MY_VAL", "SECOND_ENV" : "SECOND_VAL"})

# Multiple variables from file
fn.set_envs(file_path="env.txt")
```

## Replicas

Some runtimes can scale horizontally, configured either as a number of replicas:</br>
`spec.replicas` </br>
or a range (for auto scaling in Dask or Nuclio):</br>
```
spec.min_replicas = 1
spec.max_replicas = 4
```

```{admonition} Note
Scaling (replication) algorithm, if a `target utilization`
(Target CPU%) value is set, the replication controller calculates the utilization
value as a percentage of the equivalent `resource request` (CPU request) on
the replicas and based on that provides horizontal scaling. 
See also [Kubernetes horizontal autoscale](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/#how-does-a-horizontalpodautoscaler-work)
```

See more details in [Dask](../runtimes/dask-overview.html), [MPIJob and Horovod](../runtimes/horovod.html), [Spark](../runtimes/spark-operator.html), [Nuclio](../concepts/nuclio-real-time-functions.html).

## CPU, GPU, and memory limits for user jobs  

When you create a pod in an MLRun job or Nuclio function, the pod has default CPU and memory limits. When the job runs, it can consume 
resources up to the limits defined. The default limits are set at the service level. You can change the default limit for the service, and 
also overwrite the default when creating a job, or a function. Adding requests and limits to your function specify what compute resources 
are required. It is best practice to define this for each MLRun function.


```python
# Requests - lower bound
fn.with_requests(mem="1G", cpu=1)

# Limits - upper bound
fn.with_limits(mem="2G", cpu=2, gpus=1)
```

See more about [Kubernetes Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/).

### UI configuration
When creating a service, set the **Memory** and **CPU** in the **Common Parameters** tab, under **User jobs defaults**.
When creating a job or a function, overwrite the default **Memory**, **CPU**, or **GPU** in the **Configuration** tab, under **Resources**.

### SDK configuration

Configure the limits assigned to a function by using `with_limits`. For example:

```
training_function = mlrun.code_to_function("training.py", name="training", handler="train", 
                                           kind="mpijob", image="mlrun/mlrun-gpu")
training_function.spec.replicas = 2
training_function.with_requests(cpu=2)
training_function.with_limits(gpus=1)
```

```{admonition} Note
When specifying GPUs, MLRun uses `nvidia.com/gpu` as default GPU type. To use a different type of GPU, specify it using the optional `gpu_type` parameter.
```
## Number of workers and GPUs

For each Nuclio or serving function, MLRun creates an HTTP trigger with the default of 1 worker.  When using GPU in remote functions you must ensure 
that the number of GPUs is equal to the number of workers (or manage the GPU consumption within your code). You can set the [number of GPUs for each pod using the MLRun SDK](./create-and-use-functions.html#memory-cpu-gpu-resources).

You can change the number of workers after you create the trigger (function object), then you need to 
redeploy the function.  Examples of changing the number of workers:

{py:class}`~mlrun.runtimes.html#mlrun.runtimes.RemoteRuntime.with_http`:</br>
`serve.with_http(workers=8, worker_timeout=10)`

{py:class}`~mlrun.runtimes.html#mlrun.runtimes.RemoteRuntime.add_v3io_stream_trigger`:</br>
`serve.add_v3io_stream_trigger(stream_path='v3io:///projects/myproj/stream1', maxWorkers=3,name='stream', group='serving', seek_to='earliest', shards=1) `

## Volumes

When you create a pod in an MLRun job or Nuclio function, the pod by default has access to a file-system which is ephemeral, and gets 
deleted when the pod completes its execution. In many cases, a job requires access to files residing on external storage, or to files 
containing configurations and secrets exposed through Kubernetes config-maps or secrets.
Pods can be configured to consume the following types of volumes, and to mount them as local files in the local pod file-system:

- V3IO containers: when running on the Iguazio system, pods have access to the underlying V3IO shared storage. This option mounts a V3IO container or a subpath within it to the pod through the V3IO FUSE driver.
- PVC: Mount a Kubernetes persistent volume claim (PVC) to the pod. The persistent volume and the claim need to be configured beforehand.
- Config Map: Mount a Kubernetes Config Map as local files to the pod.
- Secret: Mount a Kubernetes secret as local files to the pod.

For each of the options, a name needs to be assigned to the volume, as well as a local path to mount the volume at (using a Kubernetes Volume Mount). Depending on the type of the volume, other configuration options may be needed, such as an access-key needed for V3IO volume.

See more about [Kubernetes Volumes](https://kubernetes.io/docs/concepts/storage/volumes/).

MLRun supports the concept of volume auto-mount which automatically mounts the most commonly used type of volume to all pods, unless disabled. See more about [MLRun auto mount](../runtimes/function-storage.html).

### UI configuration

You can configure Volumes when creating a job, rerunning an existing job, and creating an ML function.
Modify the Volumes for an ML function by pressing **ML functions**, then **<img src="../_static/images/kebab-menu.png" width="25"/>** 
of the function, **Edit** | **Resources** | **Volumes** drop-down list. 

Select the volume mount type: either Auto (using auto-mount), Manual or None. If selecting Manual, fill in the details in the volumes list 
for each volume to mount to the pod. Multiple volumes can be configured for a single pod.

### SDK configuration

Configure volumes attached to a function by using the `apply` function modifier on the function. 

For example, using v3io storage:
```
# import the training function from the Function Hub (hub://)
train = mlrun.import_function('hub://sklearn_classifier')# Import the function:
open_archive_function = mlrun.import_function("hub://open_archive")

# use mount_v3io() for iguazio volumes
open_archive_function.apply(mount_v3io())
```

You can specify a list of the v3io path to use and how they map inside the container (using volume_mounts). For example: 
```
mlrun.mount_v3io(name='data',access_key='XYZ123..',volume_mounts=[mlrun.VolumeMount("/data", "projects/proj1/data")])
```

See full details in [mount_v3io](../api/mlrun.platforms.html#mlrun.platforms.mount_v3io).

Alternatively, using a PVC volume:
```
mount_pvc(pvc_name="data-claim", volume_name="data", volume_mount_path="/data")
```

See full details in [mount_pvc](../api/mlrun.platforms.html#mlrun.platforms.mount_pvc).

## Preemption mode: Spot vs. On-demand nodes

Node selector is supported for all cloud platforms. It is relevant for MLRun and Nuclio only.

When running ML functions you might want to control whether to run on spot nodes or on-demand nodes. Preemption mode controls whether pods can be scheduled on preemptible (spot) nodes. Preemption mode is supported for all functions. 

Preemption mode uses Kubernetes Taints and Toleration to enforce the mode selected. Read more in [Kubernetes Taints and Tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration). 

### Why preemption mode

On-demand instances provide full control over the instance lifecycle. You decide when to launch, stop, hibernate, start, 
reboot, or terminate it. With Spot instances, you request capacity from specific availability zones, though it is
susceptible to spot capacity availability. This is a good choice if you can be flexible about when your applications run 
and if your applications can be interrupted. 

Here are some questions to consider when choosing the type of node:

- Is the function mission critical and must be operational at all times?
- Is the function a stateful function or stateless function?
- Can the function recover from unexpected failure?
- Is this a job that should run only when there are available inexpensive resources?

```{admonition} Important
When an MLRun job is running on a spot node and it fails, it won't get back up again. However, if Nuclio goes down due to a spot issue, it 
is brought up by Kubernetes.
```

Kubernetes has a few methods for configuring which nodes to run on. To get a deeper understanding, see [Pod Priority and Preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption).
Also, you must understand the configuration of the spot nodes as specified by the cloud provider.

### Stateless and Stateful Applications 
When deploying your MLRun jobs to specific nodes, take into consideration that on-demand 
nodes are designed to run stateful applications while spot nodes are designed for stateless applications. 
MLRun jobs are more stateful by nature. An MLRun job that is assigned to run on a spot node might be subject to interruption; 
it would have to be designed so that the job/function state will be saved when scaling to zero.

### Supported preemption modes

Preemption mode has three values:
- Allow: The function pod can run on a spot node if one is available.
- Constrain: The function pod only runs on spot nodes, and does not run if none is available. 
- Prevent: Default. The function pod cannot run on a spot node. 

To change the default function preemption mode, it is required to override mlrun the api configuration 
(and specifically "MLRUN_FUNCTION_DEFAULTS__PREENPTION_MODE" envvar to either one of the above modes).

### UI configuration

```{admonition} Note
Relevant when MLRun is executed in the [Iguazio platform](https://www.iguazio.com/docs/latest-release/).
```

You can configure Spot node support when creating a job, rerunning an existing job, and creating an ML function. 
The **Run on Spot nodes** drop-down list is in the **Resources** section of jobs. 
Configure the Spot node support for individual Nuclio functions when creating a function in the **Configuration** tab, under **Resources**. 

### SDK configuration

Configure preemption mode by adding the `with_preemption_mode` parameter in your Jupyter notebook, and specifying a mode from the list of values above. <br>
This example illustrates a function that cannot be scheduled on preemptible nodes:

```
import mlrun
import os

train_fn = mlrun.code_to_function('training', 
                            kind='job', 
                            handler='my_training_function') 
train_fn.with_preemption_mode(mode="prevent") 
train_fn.run(inputs={"dataset": my_data})
   
```

See [`with_preemption_mode`](../api/mlrun.runtimes.html#RemoteRuntime.with_preemption_mode).

Alternatively, you can specify the preemption using `with_priority_class` and `with_node_selection` parameters. This example specifies that the pod/function runs only on non-preemptible nodes:

```
import mlrun
import os
train_fn = mlrun.code_to_function('training', 
                            kind='job', 
                            handler='my_training_function') 
train_fn.with_preemption_mode(mode="prevent") 
train_fn.run(inputs={"dataset" :my_data})

fn.with_priority_class(name="default-priority")
fn.with_node_selection(node_selector={"app.iguazio.com/lifecycle":"non-preemptible"})

```

See [`with_priority_class`](../api/mlrun.runtimes.html#mlrun.runtimes.RemoteRuntime.with_priority_class).
See [`with_node_selection`](../api/mlrun.runtimes.html#mlrun.runtimes.RemoteRuntime.with_node_selection).


## Pod priority for user jobs

Pods (services, or jobs created by those services) can have priorities, which indicate the relative importance of one pod to the other pods on the node. The priority is used for 
scheduling: a lower priority pod can be evicted to allow scheduling of a higher priority pod. Pod priority is relevant for all pods created 
by the service. For MLRun, it applies to the jobs created by MLRun. For Nuclio it applies to the pods of the Nuclio-created functions.

Eviction uses these values in conjunction with pod priority to determine what to evict [Pod Priority and Preemption](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption).

Pod priority is specified through Priority classes, which map to a priority value. The priority values are: High, Medium, Low. The default is Medium. Pod priority is supported for:
- MLRun jobs: the default priority class for the jobs that MLRun creates.
- Nuclio functions: the default priority class for the user-created functions.
- Jupyter
- Presto (The pods priority also affects any additional services that are directly affected by Presto, for example like hive and mariadb, 
which are created if Enable hive is checked in the Presto service.)
- Grafana
- Shell

### UI configuration

```{admonition} Note
Relevant when MLRun is executed in the [Iguazio platform](https://www.iguazio.com/docs/latest-release/).
```
Configure the default priority for a service, which is applied to the service itself or to all subsequently created user-jobs in the 
service's **Common Parameters** tab, **User jobs defaults** section, **Priority class** drop-down list.

Modify the priority for an ML function by pressing **ML functions**, then **<img src="../_static/images/kebab-menu.png" width="25"/>** 
of the function, **Edit** | **Resources** | **Pods Priority** drop-down list.


### SDK configuration

Configure pod priority by adding the priority class parameter in your Jupyter notebook. <br>
For example:

```
import mlrun
import os
train_fn = mlrun.code_to_function('training', 
                            kind='job', 
                            handler='my_training_function') 
train_fn.with_priority_class(name={value})
train_fn.run(inputs={"dataset" :my_data})
 
```


See [with_priority_class](../api/mlrun.runtimes.html.#mlrun.runtimes.RemoteRuntime.with_priority_class).


## Node selection
Node selection can be used to specify where to run workloads (e.g. specific node groups, instance types, etc.). This is a more advanced parameter mainly used in production deployments to isolate platform services from workloads. See [**Node affinity**](../concepts/node-affinity.html) for more information on how to configure node selection.

```python
# Only run on non-spot instances
fn.with_node_selection(node_selector={"app.iguazio.com/lifecycle" : "non-preemptible"})
```

## Scaling and auto-scaling
Scaling behavior can be added to real-time and distributed runtimes including `nuclio`, `serving`, `spark`, `dask`, and `mpijob`. See [**Replicas**](./configuring-job-resources.html#replicas) to see how to configure scaling behavior per runtime. This example demonstrates setting replicas for `nuclio`/`serving` runtimes:

```python
# Nuclio/serving scaling
fn.spec.replicas = 2
fn.spec.min_replicas = 1
fn.spec.max_replicas = 4
```

## Mount persistent storage
In some instances, you might need to mount a file-system to your container to persist data. This can be done with native K8s PVC's or the V3IO data layer for Iguazio clusters. See [**Attach storage to functions**](./function-storage.html) for more information on the storage options.

```python
# Mount persistent storage - V3IO
fn.apply(mlrun.mount_v3io())

# Mount persistent storage - PVC
fn.apply(mlrun_pipelines.mounts.mount_pvc(pvc_name="data-claim", volume_name="data", volume_mount_path="/data"))
```


## Preventing stuck pods

The runtimes spec has four "state_threshold" attributes that can determine when to abort a run. 
Once a threshold is passed and the run is in the matching state, the API monitoring aborts the run, deletes its resources, 
sets the run state to aborted, and issues a "status_text" message.

The four states and their default thresholds are:

```
'pending_scheduled': '1h', #Scheduled and pending and therefore consumes resources
'pending_not_scheduled': '-1', #Scheduled but not pending, can continue to wait for resources
'image_pull_backoff': '1h', #Container running in a pod fails to pull the required image from a container registry
'running': '24h' #Job is running  
```

The thresholds are time strings constructed of value and scale pairs (e.g. "30 minutes 5h 1day"). 
To configure to infinity, use `-1`. 

To change the state thresholds, use:
```
func.set_state_thresholds({"pending_not_scheduled": "1 min"}) 
```
For just the run, use:
```
func.run(state_thresholds={"running": "1 min", "image_pull_backoff": "1 minute and 30s"}) 
```

See:
- {py:meth}`~mlrun.runtimes.KubeResource.set_state_thresholds`

```{admonition} Note
State thresholds are not supported for Nuclio/serving runtimes (since they have their own monitoring) or for the Dask runtime (which can be monitored by the client).
```

