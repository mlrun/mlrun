# Managing job resources

Configuration of job resources is relevant for all supported cloud platforms.

<!--
  ## Function!!!!!! Resources: limit and request CPU, GPU, and Memory  

  You can configure how much of each resource a function needs. Kubernetes uses this information when placing a pod on a node. The memory and 
  CPU configurations that you specify in the service are applied to each replica. Limits and requests are supported for all services.

  Service limit!

  When creating a new function, set the **Memory** and **CPU** in the **Common Parameters** tab, under **Resources** .

  Modify the Memory, CPU, GPU for an ML function by pressing **ML functions**, then press **<img src="../_static/images/kebab-menu.png" width="25"/>** 
  of the function, and select **Edit** and scroll to the **Resources** section.
-->

**In this section**
- [Node affinity (node selectors)](#node-affinity-node-selectors)
- [Preemption mode: Spot vs. On-demand nodes](#preemption-mode-spot-vs-on-demand-nodes)
- [Pod priority](#pod-priority)
- [CPU, GPU, and memory limits for user jobs](#cpu-gpu-and-memory-limits-for-user-jobs)
- [Volumes](#volumes)

## Node affinity (node selectors)

You can assign a node or a node group for services or for jobs executed by a service. When specified, the service or the pods of a function can only run on nodes whose 
labels match the node selector entries configured for the specific service. If node selection for the service is not specified, the 
selection criteria defaults to the Kubernetes default behavior, and jobs run on a random node.

For MLRun and Nuclio, you can specify node selectors on a per-job basis. The default node selectors (defined at the service level) are 
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

```func.with_node_selection(node_selector={name})```

See [with_node_selection](api/mlrun.runtimes.html?highlight=node_selector#mlrun.runtimes.RemoteRuntime.with_node_selection).


## Preemption mode: Spot vs. On-demand nodes

Preemption mode controls whether pods can be scheduled on preemptible (spot) nodes. Preemption mode is supported for all functions. Preemption mode has three values:
- Allow: The function pod can run on a spot node if one is available.
- Constrain: The function pod only runs on spot nodes, and does not run if none is available.
- Prevent: The function pod cannot run on a spot node. 

Preemption mode uses Kubernets Taints and Toleration to enforce the mode selected. Read more in [Kubernetes Taints and Tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration).


### On Demand vs. Spot 

On-demand instances provide full control over the instance lifecycle. You decide when to launch, stop, hibernate, start, 
reboot, or terminate it. With Spot instances you request capacity from specific availability zones, though it is  
susceptible to spot capacity availability. This is a good choice if you can be flexible about when your applications run 
and if your applications can be interrupted. 

### Stateless and Stateful Applications 
When deploying your MLRun jobs to specific nodes, take into consideration that On-demand 
nodes are designed to run stateful applications while spot nodes are designed for stateless applications. 
MLRun jobs that are stateful and are assigned to run on spot nodes might be subject to interruption 
and have to be designed so that the job/function state will be saved when scaling to zero.

### UI configuration

```{admonition} Note
Relevant when MLRun is executed in the [Iguazio platform](https://www.iguazio.com/docs/latest-release/).
```

You can configure Spot node support when creating a job, rerunning an existing job, and creating an ML function. 
The **Run on Spot nodes** drop-down list is in the **Resources** section of jobs. 
Configure the Spot node support for individual Nuclio functions when creating a function in the **Configuration** tab, under **Resources**. 

### SDK configuration

Configure preemption mode by adding the parameter in your Jupyter notebook, specifying a mode from the list of values above. <br>
For example:

```func.with_preemption_mode(mode={value})```

See [with_node_selection](api/mlrun.runtimes.#mlrun.runtimes.RemoteRuntime.with_node_selection).


## Pod priority

Pods (services, or jobs created by those services) can have priorities, which indicate the relative importance of one pod to the other pods on the node. The oriority is used for 
scheduling: a lower priority pod can be evicted to allow scheduling of a higher priority pod. Pod priority is relevant for all pods created 
by the service. For MLRun, it applies to the jobs created by MLRun. For Nuclio it applies to the pods of the Nuclio-created functions.

Eviction uses these values to determine what to evict with conjunction to the pods priority [Interactions between Pod priority and quality of service](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption/#interactions-of-pod-priority-and-qos).

Pod priority is specified through Priority classes, which map to a priority value. The priority values are: High, Medium, Low. The default is Medium. Pod priority is supported for:
- MLRun jobs: the default priority class for the jobs that MLRun creates.
- Nuclio functions: the default priority class for the user-created functions.
- Jupyter
- Presto (The pods priority also affects any additional services that are directly affected by Presto, for example like hive and mariadb, 
which are created if Enable hive is checked in the Presto service.)
- Grafana
- Shell

See more about [Kubernetes Pod Priority and Preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)

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

```func.with_priority_class(name={value})```


See [with_priority_class](api/mlrun.runtimes.html.#mlrun.runtimes.RemoteRuntime.with_priority_class).

## CPU, GPU, and memory limits for user jobs  

When you create a pod in an MLRun job or Nuclio function, the pod has default CPU and memory limits. When the job runs, it can consume 
resources up to the limits defined. The default limits are set at the service level. You can change the default limit for the service, and 
also overwrite the default when creating a job, or a function. 
<!-- The default values are: -->

See more about [Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/).

### UI configuration
When creating a service, set the **Memory** and **CPU** in the **Common Parameters** tab, under **User jobs defaults**.
When creating a job or a function, overwrite the default **Memory**, **CPU**, or **GPU** in the **Configuration** tab, under **Resources**.

### SDK configuration



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

See more about [Volumes](https://kubernetes.io/docs/concepts/storage/volumes/).

MLRun supports the concept of volume auto-mount which automatically mounts the most commonly used type of volume to all pods, unless disabled. See more about [MLRun auto mount](../runtimes/function-storage.html).

### UI configuration

You can configure Volumes when creating a job, rerunning an existing job, and creating an ML function.
Modify the Volumes for an ML function by pressing **ML functions**, then **<img src="../_static/images/kebab-menu.png" width="25"/>** 
of the function, **Edit** | **Resources** | **Volumes** drop-down list. 

Select the volume mount type: either Auto (using auto-mount), Manual or None. If selecting Manual, fill in the details in the volumes list .for each volume to mount to the pod. Multiple volumes can be configured for a single pod.

### SDK configuration

Configure Volumes attached to a function by using the `apply` function modifier on the function. For example:

```func.apply(mlrun.platforms.mount_v3io())```

See [list of MLRun mount modifiers](../api/mlrun.platforms.html).





