(distributed-functions)=
# Distributed runtimes

Many of the runtimes support horizontal scaling. You can specify the number of `replicas` or the 
min&mdash;max value range (for auto scaling in Dask or Nuclio). When scaling functions MLRun uses a high-speed
messaging protocol and shared storage (volumes, objects, databases, or streams). MLRun runtimes
handle the orchestration and monitoring of the distributed task.

<img src="../_static/images/runtime-scaling.png" alt="runtime-scaling" width="600"/><br>

**In this section**
```{toctree}
:maxdepth: 1

../runtimes/dask-overview
../runtimes/horovod
../runtimes/spark-operator
```