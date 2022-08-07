(node-affinity)=
# Node affinity

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

## UI configuration

Configure node selection on the service level in the service's **Custom Parameters** tab, under **Resources**, by adding or removing 
Key:Value pairs. For MLRun and Nuclio, this is the default node selection for all MLRun jobs and Nuclio functions. 

You can also configure the node selection for individual MLRun jobs by going to **Platform dashboard | Projects | New Job | Resources | Node 
selector**, and adding or removing Key:Value pairs. Configure the node selection for individual Nuclio functions when creating a function in 
the **Confguration** tab, under **Resources**, by adding Key:Value pairs.

## SDK configuration

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
