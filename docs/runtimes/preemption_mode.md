# Preemption modes

```{admonition} Note
This section is relavant to cloud environment only
```

When running ML functions there are times when we want to control whether to run on spot nodes or on-demand nodes.
**In this section**
- [Considerations for running on Preemptible nodes](#Considerations-for-running-on-Preemptible-nodes)
- [Why preemption modes](#Why-preemption-modes)
- [Supported preemption modes](#Supported-preemption-modes)
- [How to set preemption mode](#How-to-set-preemption-mode)


## Considerations for running on preemptible nodes
Here are some questions to consider while choosing on the type of nodes to use:

- Is the function mission critical and must be operational at all times?
- Is the function is statefull function or stateless function?
- Can the function recover from unexpected failure?
- Is the work an opportunist job in which you run if inexpensive resources are available?

## Why preemption modes
Kuberenetes has a few methods for configuring which nodes to run on, but it takes a deeper understanding of several components in kuberenetes as well as the configuration of the spot nodes specified by the cloud provider.

```{admonition} Note
💡 MLRun supports setting node related kuberenetes configuration.
Read more in [https://docs.mlrun.org/en/latest/runtimes/node-affinity.html#node-affinity-for-mlrun-jobs](https://docs.mlrun.org/en/latest/runtimes/node-affinity.html#node-affinity-for-mlrun-jobs) .
```
## Supported preemption modes
With the introduction of preemption modes, we want to eliminate the need to understand the underlying setup necessary for kuberenetes to execute on various types of nodes.

The supported modes are :

- **allow** - The function can be scheduled on both on-demand and preemptible nodes
- **constrain** - The function can only run on preemptible nodes.
- **prevent** - The function cannot be scheduled on preemptible nodes and will only run on on-demand nodes.
- **none** - No preemptible configuration will be applied on the function.

```{admonition} Note
The default preemption mode is set to **prevent**.
<br>
The default preemption mode is configurable in ```mlrun.mlconf.function_defaults.preemption_mode```.
<br>
```
## How to set preemption mode
To set preemption mode with using the sdk :
<br><br>
```func.with_preemption_mode(mode=mlrun.schemas.PreemptionModes.allow)```

or 

```func.with_preemption_mode(mode="allow")```
<br><br>