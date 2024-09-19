(configuring-job-resources)=
# Configuring runs and functions

MLRun orchestrates serverless functions over Kubernetes. You can specify the resource requirements (CPU, memory, GPUs),
preferences, and pod priorities in the logical function object. You can also configure how MLRun prevents stuck pods.
All of these are used during the function deployment.

Configuring runs and functions is relevant for all supported cloud platforms.

**In this section**
- [Environment variables](#environment-variables)
- [Replicas](#replicas)
- [CPU, GPU, and memory &mdash; requests and limits for user jobs](#cpu-gpu-and-memory-requests-and-limits-for-user-jobs)
- [Number of workers and GPUs](#number-of-workers-and-gpus)
- [Volumes](#volumes)
- [Preemption mode: Spot vs. On-demand nodes](#preemption-mode-spot-vs-on-demand-nodes)
- [Pod priority for user jobs](#pod-priority-for-user-jobs)
- [Node selection](#node-selection)
- [Scaling and auto-scaling](#scaling-and-auto-scaling)
- [Mounting persistent storage](#mounting-persistent-storage)
- [Preventing stuck pods](#preventing-stuck-pods)
- [Setting the log level](#setting-the-log-level)

## Environment variables

Environment variables can be added individually, from a Python dictionary, or a file:

```python
# Single variable
fn.set_env(name="MY_ENV", value="MY_VAL")

# Multiple variables
fn.set_envs(env_vars={"MY_ENV": "MY_VAL", "SECOND_ENV": "SECOND_VAL"})

# Multiple variables from file
fn.set_envs(file_path="env.txt")
```

## Replicas

Some runtimes can scale horizontally, configured either as a number of replicas:
```python
training_function = mlrun.code_to_function(
    "training.py",
    name="training",
    handler="train",
    kind="mpijob",
    image="mlrun/mlrun-gpu",
)
training_function.spec.replicas = 2
```
or a range (for auto-scaling in Dask or Nuclio):

```
# set range for # of replicas with replicas and max_replicas
dask_cluster.spec.min_replicas = 1
dask_cluster.spec.max_replicas = 4
```

```{admonition} Note
If a `target utilization`
(Target CPU%) value is set, the replication controller calculates the utilization
value as a percentage of the equivalent `resource request` (CPU request) on
the replicas and based on that provides horizontal scaling. 
See also [Kubernetes horizontal autoscale](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/#how-does-a-horizontalpodautoscaler-work).
```

See more details in [Dask](../runtimes/dask-overview.html), [MPIJob and Horovod](../runtimes/horovod.html), [Spark](../runtimes/spark-operator.html), [Nuclio](../concepts/nuclio-real-time-functions.html).

## CPU, GPU, and memory &mdash; requests and limits for user jobs

Requests and limits define how much the memory, CPU, and GPU, the pod must have to be able to start to work, and its maximum allowed consumption.
MLRun and Nuclio functions run in their own pods. The default CPU and memory limits for these pods are defined by their respective services. 
You can change the limits when creating a job, or a function. It is best practice to define this for each MLRun function. 

See more details in the [Kubernetes documentation: Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/).

### SDK configuration

Examples of {py:meth}`~mlrun.runtimes.KubeResource.with_requests` and  {py:meth}`~mlrun.runtimes.KubeResource.with_limits`:

```python
training_function = mlrun.code_to_function(
    "training.py",
    name="training",
    handler="train",
    kind="mpijob",
    image="mlrun/mlrun-gpu",
)
training_function.with_requests(mem="1G", cpu=1)  # lower bound
training_function.with_limits(mem="2G", cpu=2, gpus=1)  # upper bound
```

```{admonition} Note
When specifying GPUs, MLRun uses `nvidia.com/gpu` as default GPU type. To use a different type of GPU, specify it using the optional `gpu_type` parameter.
```

### UI configuration

```{admonition} Note
Relevant when MLRun is executed in the [Iguazio platform](https://www.iguazio.com/docs/latest-release/).
```
Configure requests and limits in the service's **Common Parameters** tab and in the **Configuration** tab of the function.

## Number of workers and GPUs

For each Nuclio or serving function, MLRun creates an HTTP trigger with the default of 1 worker.  When using GPU in remote functions you must ensure 
that the number of GPUs is equal to the number of workers (or manage the GPU consumption within your code). You can set the [number of GPUs for each pod using the MLRun SDK](#cpu-gpu-and-memory-requests-and-limits-for-user-jobs).

You can change the number of workers after you create the trigger (function object), then you need to 
redeploy the function. Examples of changing the number of workers:

using {py:meth}`mlrun.runtimes.RemoteRuntime.with_http`:</br>
`serve.with_http(workers=8, worker_timeout=10)`

using {py:meth}`mlrun.runtimes.RemoteRuntime.add_v3io_stream_trigger`:</br>
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

MLRun supports the concept of volume auto-mount, which automatically mounts the most commonly used type of volume to all pods, unless disabled. See more about [MLRun auto mount](../runtimes/function-storage.html).

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

See full details in {py:meth}`~mlrun.platforms.mount_v3io`.

Alternatively, using a PVC volume:
```
mount_pvc(pvc_name="data-claim", volume_name="data", volume_mount_path="/data")
```

See full details in {py:meth}`~mlrun.platforms.mount_pvc`.
### UI configuration

```{admonition} Note
Relevant when MLRun is executed in the [Iguazio platform](https://www.iguazio.com/docs/latest-release/).
```
You can configure Volumes when creating a job, rerunning an existing job, and creating an ML function.
Modify the Volumes for an ML function by pressing **ML functions**, then **<img src="../_static/images/kebab-menu.png" width="25"/>** 
of the function, **Edit** | **Resources** | **Volumes** drop-down list. 

Select the volume mount type: either Auto (using auto-mount), Manual or None. If selecting Manual, fill in the details in the volumes list 
for each volume to mount to the pod. Multiple volumes can be configured for a single pod.


## Preemption mode: Spot vs. On-demand nodes
When running ML functions you might want to control whether to run on spot nodes or on-demand nodes. Preemption mode controls 
whether pods can be scheduled on preemptible (spot) nodes. Preemption mode is supported for all functions. 

Preemption mode uses [Kubernetes Taints and Tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration) to enforce the mode selected.  

### Why preemption mode?

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

Kubernetes has a few methods for configuring which nodes to run on. To get a deeper understanding, see 
[Pod Priority and Preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption).
Also, you must understand the configuration of the spot nodes as specified by the cloud provider.

### Stateless and Stateful Applications 
When deploying your MLRun jobs to specific nodes, take into consideration that on-demand 
nodes are designed to run stateful applications while spot nodes are designed for stateless applications. 
MLRun jobs are more stateful by nature. An MLRun job that is assigned to run on a spot node might be subject to interruption; 
it would have to be designed so that the job/function state will be saved when scaling to zero.

### Supported preemption modes

Preemption mode has these values:
- allow: The function pod can run on a spot node if one is available.
- constrain: The function pod only runs on spot nodes, and does not run if none is available. 
- prevent: Default. The function pod cannot run on a spot node. 
- none: No preemptible configuration is applied to the function

To change the default function preemption mode, it is required to override the api configuration 
(and specifically "MLRUN_FUNCTION_DEFAULTS__PREEMPTION_MODE" environment variable to either one of the above modes).

### SDK configuration

Configure preemption mode by adding the {py:meth}`~mlrun.runtimes.KubeResource.with_preemption_mode` parameter in your Jupyter notebook,  specifying a mode from the list of values above. <br>
This example illustrates a function that cannot be scheduled on preemptible nodes:


```
# Can be scheduled on a preemptible (spot) node
fn. with_preemption_mode("allow")
```

And another function that can only be scheduled on preemptible nodes:

```
import mlrun
import os

train_fn = mlrun.code_to_function('training', 
                            kind='job', 
                            handler='my_training_function') 
train_fn.with_preemption_mode(mode="prevent") 
train_fn.run(inputs={"dataset": my_data})
```

See {py:meth}`~KubeResource.with_preemption_mode.

Alternatively, you can specify the preemption using `with_priority_class` and `with_node_selection` parameters. This example specifies that 
the pod/function runs only on non-preemptible nodes:

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

See:
- {py:meth}`~mlrun.runtimes.KubeResource.with_priority_class`
- {py:meth}`~mlrun.runtimes.KubeResource.with_node_selection`

### UI configuration

```{admonition} Note
Relevant when MLRun is executed in the [Iguazio platform](https://www.iguazio.com/docs/latest-release/).
```

You can configure Spot node support when creating a job, rerunning an existing job, and creating an ML function. 
The **Run on Spot nodes** drop-down list is in the **Resources** section of jobs. 
Configure the Spot node support for individual Nuclio functions when creating a function in the **Configuration** tab, under **Resources**. 

## Pod priority for user jobs

[Priority classes](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/) are a mechanism in Kubernetes to 
control the order in which pods are scheduled and evicted &mdash; to make room for other, higher priority pods. Priorities also affect the pods’ 
evictions in case the node’s memory is pressured (called Node-pressure Eviction).

Pod priority is relevant for all of the jobs created by MLRun. For Nuclio it applies to the pods of the Nuclio-created functions.

Pod priority is specified through Priority classes, which map to a priority value. Use these to view the priority classes and the default:
- `fn.list_valid_priority_class_names()`
- `fn.get_default_priority_class_name()`

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

See {py:meth}`~mlrun.runtimes.KubeResource.with_priority_class`.

### UI configuration

```{admonition} Note
Relevant when MLRun is executed in the [Iguazio platform](https://www.iguazio.com/docs/latest-release/).
```
Configure the default priority for a service, which is applied to the service itself or to all subsequently created user-jobs in the 
service's **Common Parameters** tab, **User jobs defaults** section, **Priority class** drop-down list.

Modify the priority for an ML function by pressing **ML functions**, then **<img src="../_static/images/kebab-menu.png" width="25"/>** 
of the function, **Edit** | **Resources** | **Pods Priority** drop-down list.

## Node selection

```{admonition} Note
Requires Nuclio v1.13.5 or higher.
```
Node selection can be used to specify where to run workloads (e.g. specific node groups, instance types, etc.). This is a more advanced 
parameter mainly used in production deployments to isolate platform services from workloads. You can assign a node or a node group for MLRun or Nuclio service, 
for jobs executed by a service, and at the project level.  When specified, the 
service/function/project can only run on nodes whose labels match the node selector entries configured for the specific service/function/project. 

Configurations at the project and function levels are treated as a cohesive unit, prioritizing the function level. 
Therefore, configurations defined at the function level take precedence over those at the project level. 
Configurations set at either the project or function level (or both) take precedence over 
those at the service level: if any configuration is specified at the project or function level (or both), the service level 
configuration is not considered.   

If node selection is not specified, the selection criteria defaults to the Kubernetes default behavior: the service/function run on a random node.

To illustrate this logic, consider the following cases:

- MLRun service level (V), project level (V), function level (X): Merge between service and project levels, with project-level configuration taking precedence.
- MLRun service level (V), project level (X), function level (V): Merge between service and function levels, with function-level configuration taking precedence.
- MLRun service level (V), project level (V), function level (V): Merge between service, project and function levels, with function-level configuration taking precedence.
- MLRun service level (V), project level (X), function level (X): Service-level configuration applies to the function.

Here's an example that demonstrates how the function-level configuration overrides the project-level configuration, 
while still incorporating any additional labels defined at the service level:
- The service level defines node selectors like `{"region": "us-central1", "gpu": "False", "arch": "arm64"}`,
- The project level defines node selectors like `{"zone": "us-west1", "arch": "amd64"}`,
- The function level specifies `{"zone": "us-east1", "gpu": "true"}`.

The resulting configuration for the function is:
```python
{"region": "us-central1", "zone": "us-east1", "gpu": "true", "arch": "amd64"}
```

### Overriding node selectors

You can override and ignore node selectors defined at the project level or service level from the function level 
by using an empty key (a key with no value), thereby completely canceling a specific node selector label. For example, if:
- The project level defines `{"zone": "us-west1", "arch": "amd64"}`
- The function level specifies `{"zone": "", "gpu": "true"}`

The zone label from the project level is completely removed, and the resulting configuration for the function is:
```
{"gpu": "true", "arch": "amd64"}
```



### Runtimes
Each runtime type is handled individually, with specific behaviors defined for Nuclio and Spark. These special behaviors ensure 
that each runtime type is handled according to its unique requirements.

- Nuclio: For all runtime types, the node selector is applied to the run object that was created as a result of the execution.
Since Nuclio doesn't have a runtime object in the same way as other runtimes, the final merged node selector (derived 
from the MLRun config level, project level, and function level) is passed directly to the Nuclio config.
This merged node selector becomes the function configuration for Nuclio, and Nuclio itself performs a similar operation, 
merging it with the Nuclio service level config.
The result is that the MLRun service configuration has precedence over the Nuclio service config. However, if there is no overlap 
in the labels, both are reflected in the final output.

- Spark: Spark has three separate node selector settings: `application_node_selector`, `driver_node_selector`, and `executor_node_selector`. 
When setting a node selector for the application, it only applies to the driver and executor, as there is no real 
significance to setting it for the application itself (since the only pods created are for the driver and executor). 
The logic is:
   - Application Node Selector: Always remains empty.
   - Driver Node Selector: If no specific `driver_node_selector` is defined, the runtime node selector is used. 
If a specific `driver_node_selector` is defined, it takes precedence. After selecting the appropriate driver node selector, 
a merge with precedence is performed with the project and MLRun config levels.
   - Executor Node Selector: Follows the same logic as the driver node selector. If no specific `executor_node_selector` is defined, 
the runtime node selector is used. If a specific `executor_node_selector` is defined, it takes precedence. 
A merge with precedence is then performed with the project and MLRun config levels.

This logic becomes part of the Spark CRD, ensuring that it is consistently applied during the job execution. 

### Best Practice

Node selection is often used for assigning jobs/pods to GPU nodes. But not all jobs/pods benefit from a GPU node.
For example, a Databricks “helper” pod runs in a Spark service on Databricks and doesn’t follow the node-selector 
(and doesn't benefit from being assigned to a GPU node). 
A Spark Function includes an executor and a driver; the driver also does not benefit from a GPU node.

### SDK configuration

Configure node selection by adding the key:value pairs in your Jupyter notebook formatted as a Python dictionary. <br>
For example:

```        
# Run a function only on non-spot instances
fn.with_node_selection(node_selector={"app.iguazio.com/lifecycle" : "non-preemptible"})
```
```
# Run a project on specific instances
project.with_node_selection(node_selector={"zone": "us-west1"})
```

```
# Cancel a node selector
fn.with_node_selection(node_selector={"zone": })
```

See {py:meth}`~mlrun.runtimes.RemoteRuntime.with_node_selection`.

### UI configuration
```{admonition} Note
Relevant when MLRun is executed in the [Iguazio platform](https://www.iguazio.com/docs/latest-release/).
```
- Configure node selection for individual MLRun jobs when creating a Batch run by going to **Platform dashboard | Projects | New Job | Resources | Node selector**, 
and adding or removing Key:Value pairs. 
- Configure the node selection for individual Nuclio functions when creating a 
function in the **Confguration** tab, under **Resources**, by adding **Key:Value** pairs.
- Configure node selection on the function level in the **Projects | <project> | Settings**, by adding or removing 
Key:Value pairs.

## Scaling and auto-scaling
Scaling behavior can be added to real-time and distributed runtimes including `nuclio`, `serving`, `spark`, `dask`, and `mpijob`. 
In environments where node auto-scaling is available, auto-scaling is triggered in situations where pods cannot be scheduled to any existing node 
due to lack of resources. In situations where pod requests for CPU/Memory are low, auto-scaling may not be triggered since pods could still be 
placed on existing nodes (per their low requests), even though in practice they do not have the needed resources as they near their (much higher) 
limits and might be in danger of eviction due to OOM situations.

Auto-scaling works best when jobs are created with limit=request. In this situation, once resources are not sufficient, new jobs are not 
scheduled to any existing node, and new nodes are automatically added to accommodate them.

Auto-scaling is a node-group configuration.

## Mounting persistent storage
In some instances, you might need to mount a file-system to your container to persist data. This can be done with native K8s PVC's or the V3IO data layer for Iguazio clusters. See [**Attach storage to functions**](./function-storage.html) for more information on the storage options.

```python
# Mount persistent storage - V3IO
fn.apply(mlrun.mount_v3io())

# Mount persistent storage - PVC
fn.apply(
    mlrun.platforms.mount_pvc(
        pvc_name="data-claim", volume_name="data", volume_mount_path="/data"
    )
)
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
```python
func.set_state_thresholds({"pending_not_scheduled": "1 min"})
```
For just the run, use:
```python
func.run(
    state_thresholds={"running": "1 min", "image_pull_backoff": "1 minute and 30s"}
)
```

See {py:meth}`~mlrun.runtimes.KubeResource.set_state_thresholds`

```{admonition} Note
State thresholds are not supported for Nuclio/serving runtimes (since they have their own monitoring) or for the Dask runtime (which can be monitored by the client).
```

## Setting the log level

You can set the log level for individual functions. 

To set the log level in the function itself: `context.logger.set_logger_level(level="WARN")`

To set the log level outside the function, using an environment variable: `func.set_env(name="MLRUN_LOG_LEVEL",value="WARN")`

To set the log level for a Nuclio function (Remote, Serving or Application runtime): `func.set_config(key="spec.loggerSinks", value=[{"level":"warning"}])`

