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

## Node affinity (node selectors)

You can assign a node or a node group for jobs executed by a service. When specified, the pods of a function can only run on nodes whose 
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

Configure node selection by adding the key:value pairs in your Jupyter notebook. <br>
For example:

```func.with_node_selection(node_selector={name})```

See [with_node_selection](api/mlrun.runtimes.html?highlight=node_selector#mlrun.runtimes.RemoteRuntime.with_node_selection).


## Pod toleration (Spot vs. On-demand nodes)

Pod toleration controls whether pods can be scheduled on spot nodes. Pod toleration is supported for all functions. Run on Spot nodes has three values:
- Allow: The function pod can run on a spot node if one is available.
- Constrain: The function pod only runs on spot nodes, and does not run if none is available.
- Prevent: The function pod cannot run on a spot node. 

See more about [Kubernetes Taints and Tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration).


### On Demand vs. Spot 

On-demand instances provide full control over the instance lifecycle. You decide when to launch, stop, hibernate, start, 
reboot, or terminate it. With Spot instances you request capacity from specific availability zones, though it is  
susceptible to spot capacity availability. This is a good choice if you can be flexible about when your applications run 
and if your applications can be interrupted. 

### Stateless and Stateful Applications 
When deploying your MLRun jobs to specific nodes, take into consideration that On-demand 
nodes are designed to run stateful applications while spot nodes are designed for stateless applications. 
MLRun jobs that are stateful and are assigned to run on spot nodes, might be subject to interruption 
and will to be designed so that the job/function state will be saved when scaling to zero.

### UI configuration

```{admonition} Note
Relevant when MLRun is executed in the [Iguazio platform](https://www.iguazio.com/docs/latest-release/).
```

You can configure pod toleration when creating a job, rerunning an existing job, and creating an ML function. 
The **Run on Spot nodes** drop-down list is in the **Resources** section of jobs. 
Configure the pod toleration for individual Nuclio functions when creating a function in the **Configuration** tab, under **Resources**. 

### SDK configuration

Configure pod toleration by adding the tolerations parameter in your Jupyter notebook. <br>
For example:

```func.with_preemption_mode(mode={value})```

See [with_node_selection](api/mlrun.runtimes.#mlrun.runtimes.RemoteRuntime.with_node_selection).


## Pod priority

Pods can have priorities, which indicate the relative importance of one pod to the other pods on the node. The oriority is used for 
scheduling: a lower priority pod can be evicted to allow scheduling of a higher priority pod. Pod priority is relevant for all pods created 
by the service. For MLRun, it applies to the jobs created by MLRun. For Nuclio it applies to the pods of the Nuclio-created functions.

Eviction uses these values to determine what to evict with conjunction to the pods priority [Interactions between Pod priority and quality of service](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption/#interactions-of-pod-priority-and-qos).

The priority values are: High, Medium, Low. The default is Medium. Pod priority is supported for:
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
Configure the default priority for a service, which is applied to all subsequently created user-jobs in the service's **Common Parameters** 
tab, **User jobs defaults** section, **Priority class** drop-down list.

Modify the priority for an ML function by pressing **ML functions**, then **<img src="../_static/images/kebab-menu.png" width="25"/>** 
of the function, **Edit** | **Resources** | **Pods Priority** drop-down list.


### SDK configuration

Configure pod priority by adding the priority class parameter in your Jupyter notebook. <br>
For example:

```func.with_priority_class(name={value})```


See [with_priority_class](api/mlrun.runtimes.html.#mlrun.runtimes.RemoteRuntime.with_priority_class).

## CPU, GPU, and memory limits for user jobs  

When you create a pod in an MLRun job or Nuclio function, the pod has default CPU and memory limits. When the job runs, it allocates 
resources for itself starting with the default values. The default is set at the service level. You can overwrite the default when creating a job, or a function. 
<!-- The default values are: -->

See more about [Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/).

### UI configuration
When creating a service, set the **Memory** and **CPU** in the **Common Parameters** tab, under **User jobs defaults**.
When creating a job or a function, overwrite the default **Memory**, **CPU**, or **GPU** in the **Configuration** tab, under **Resources**.

### SDK configuration



## Volumes

### UI configuration

### SDK configuration








