# Managing job resources

MLRun orchestrates serverless functions over Kubernetes. You can specify the desired resource requirements (CPU, memory, GPUs) and preferences/priorities in the 
logical function object. These are used during the function deployment.

Configuration of job resources is relevant for all supported cloud platforms.

**In this section**
- [Replicas](#replicas)
- [CPU, GPU, and memory limits for user jobs](#cpu-gpu-and-memory-limits-for-user-jobs)
- [Volumes](#volumes)
- [Preemption mode: Spot vs. On-demand nodes](#preemption-mode-spot-vs-on-demand-nodes)
- [Pod priority](#pod-priority-for-user-jobs)
- [Node affinity (node selectors)](#node-affinity-node-selectors)

## Replicas

Some runtimes can scale horizontally, configured either as a number of replicas:</br>
`spec.replicas` </br>
or a range (for auto scaling in Dask or Nuclio:</br>
```
spec.min_replicas = 1
spec.max_replicas = 4
```
See more details in [Dask](../runtimes/dask-overview), [MPIJob and Horovod](../runtimes/horovod), [Spark](../runtimes/spark-operator), [Nuclio](../concepts/nuclio-real-time-functions).

## CPU, GPU, and memory limits for user jobs  

When you create a pod in an MLRun job or Nuclio function, the pod has default CPU and memory limits. When the job runs, it can consume 
resources up to the limits defined. The default limits are set at the service level. You can change the default limit for the service, and 
also overwrite the default when creating a job, or a function. 

See more about [Kubernetes Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/).

### UI configuration
When creating a service, set the **Memory** and **CPU** in the **Common Parameters** tab, under **User jobs defaults**.
When creating a job or a function, overwrite the default **Memory**, **CPU**, or **GPU** in the **Configuration** tab, under **Resources**.

### SDK configuration

Configure the limits assigned to a function by using `with_limits`. For example:

```
training_function = mlrun.code_to_function("training.py", name="training", handler="train", 
                                                                       kind="mpijob", image="mlrun/ml-models-gpu")
training_function.spec.replicas = 2
training_function.with_requests(cpu=2)
training_function.gpus(1)
```

```{admonition} Note
When specifying GPUs, MLRun uses `nvidia.com/gpu` as default GPU type. To use a different type of GPU, specify it using the optional `gpu_type` parameter.
```

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

MLRun supports the concept of volume auto-mount which automatically mounts the most commonly used type of volume to all pods, unless disabled. See more about [MLRun auto mount](function-storage).

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
# import the training function from the marketplace (hub://)
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

Preemption mode uses Kubernets Taints and Toleration to enforce the mode selected. Read more in [Kubernetes Taints and Tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration). 

### Why preemption mode

On-demand instances provide full control over the instance lifecycle. You decide when to launch, stop, hibernate, start, 
reboot, or terminate it. With Spot instances you request capacity from specific availabile zones, though it is  
susceptible to spot capacity availability. This is a good choice if you can be flexible about when your applications run 
and if your applications can be interrupted. 

Here are some questions to consider when choosing the type of node:

- Is the function mission critical and must be operational at all times?
- Is the function a stateful function or stateless function?
- Can the function recover from unexpected failure?
- Is this a job that should run only when there are available inexpensive resources?

```{admonition} Important
When an MLRun job is running on a spot node and it fails, it won't get back up again. Howewer, if Nuclio goes down due to a spot issue, it 
is brought up by Kubernetes.
```

Kuberenetes has a few methods for configuring which nodes to run on. To get a deeper understanding , see [Pod Priority and Preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption).
Also, you must understand the configuration of the spot nodes as specified by the cloud provider.

### Stateless and Stateful Applications 
When deploying your MLRun jobs to specific nodes, take into consideration that on-demand 
nodes are designed to run stateful applications while spot nodes are designed for stateless applications. 
MLRun jobs are more stateful by nature. An MLRun job that is assigned to run on a spot node might be subject to interruption; 
it would have to be designed so that the job/function state will be saved when scaling to zero.

## Supported preemption modes

Preemption mode has three values:
- Allow: The function pod can run on a spot node if one is available.
- Constrain: The function pod only runs on spot nodes, and does not run if none is available.
- Prevent: Default. The function pod cannot run on a spot node. 

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
train_fn.run(inputs={"dataset" :my_data})
   
```

See [`with_preemption_mode`](../api/mlrun.runtimes.html#RemoteRuntime.with_preemption_mode).

Alternatively, you can specify the preemption using `with_priority_class` and `fn.with_priority_class(name="default-priority")node_selector`. This example specifies that the pod/function runs only on non-preemptible nodes:

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
                           
See [`with_node_selection`](../api/mlrun.runtimes.html#mlrun.runtimes.RemoteRuntime.with_node_selection).


## Pod priority for user jobs

Pods (services, or jobs created by those services) can have priorities, which indicate the relative importance of one pod to the other pods on the node. The priority is used for 
scheduling: a lower priority pod can be evicted to allow scheduling of a higher priority pod. Pod priority is relevant for all pods created 
by the service. For MLRun, it applies to the jobs created by MLRun. For Nuclio it applies to the pods of the Nuclio-created functions.

Eviction uses these values to determine what to evict with conjunction to the pods priority [Pod Priority and Preemption](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption).

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

## Node affinity (node selectors)

You can assign a node or a node group for services or for jobs executed by a service. When specified, the service or the pods of a function can only run on nodes whose 
labels match the node selector entries configured for the specific service. If node selection for the service is not specified, the 
selection criteria defaults to the Kubernetes default behavior, and jobs run on a random node.

For MLRun and Nuclio, you can also specify node selectors on a per-job basis. The default node selectors (defined at the service level) are 
applied to all jobs unless you specifically override them for an individual job. 

You can configure node affinity for:
- Jupyter
- Presto (The node selection also affects any additional services that are directly affected by Presto, for example hive and mariadb, 
which are created if Enable hive is checked in the Presto service.)
- Grafana
- Shell
- MLRun (default value applied to all jobs that can be overwritten for individual jobs)
- Nuclio (default value applied to all jobs that can be overwritten for individual jobs)

See more about [Kubernetes nodeSelector](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#nodeselector).

### UI configuration

Configure node selection on the service level in the service's **Custom Parameters** tab, under **Resources**, by adding or removing 
Key:Value pairs. For MLRun and Nuclio, this is the default node selection for all MLRun jobs and Nuclio functions. 

You can also configure the node selection for individual MLRun jobs by going to **Platform dashboard | Projects | New Job | Resources | Node 
selector**, and adding or removing Key:Value pairs. Configure the node selection for individual Nuclio functions when creating a function in 
the **Confguration** tab, under **Resources**, by adding Key:Value pairs.

### SDK configuration

Configure node selection by adding the key:value pairs in your Jupyter notebook formatted as a Python dictionary. <br>
For example:

```
import mlrun
import os
train_fn = mlrun.code_to_function('training', 
                            kind='job', 
                            handler='my_training_function') 
train_fn.with_preemption_mode(mode="prevent") 
train_fn.run(inputs={"dataset" :my_data})

            
# Add node selection
func.with_node_selection(node_selector={name})
```

See [`with_node_selection`](../api/mlrun.runtimes.html#mlrun.runtimes.RemoteRuntime.with_node_selection).
