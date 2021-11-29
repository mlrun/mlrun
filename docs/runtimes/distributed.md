
# Distributed and Hyper-param Jobs

Many of the runtimes support horizontal scaling, you can specify the number of `replicas` or the 
min - max value range (for auto scaling in Dask or Nuclio). When scaling functions we use some high speed
messaging protocol and shared storage (volumes, objects, databases, or streams). MLRun runtimes
handle the orchestration and monitoring of the distributed task.

<img src="../_static/images/runtime-scaling.png" alt="runtime-scaling" width="600"/><br>

MLRun also support iterative tasks for automatic and distributed execution of many tasks with variable parameters (hyper params)
, the iterative tasks can be distributed across multiple containers and allow hyper-parameter training, AutoML, and other parallel tasks.

**For more details see:**

```{toctree}
:maxdepth: 1
../hyper-params
dask-overview
horovod
spark-operator
```
