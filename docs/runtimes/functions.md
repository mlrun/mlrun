(Functions)=
# Serverless functions

All the executions in MLRun are based on **Serverless functions**. The functions allow specifying code and 
all the operational aspects (image, required packages, [cpu/mem/gpu resources](../runtimes/configuring-job-resources.html#cpu-gpu-and-memory-limits-for-user-jobs), [storage](../runtimes/function-storage.html), environment, etc.). 
The [different function runtimes](../concepts/functions-overview.html) take care of automatically transforming the code and spec to fully 
managed and elastic services over Kubernetes, which saves significant operational overhead, 
addresses scalability and reduces infrastructure costs.

MLRun supports:
- Real-time functions for: {ref}`serving <serving-graph>`, APIs, and stream processing (based on the high-performance {ref}`Nuclio <nuclio-real-time-functions>` engine). 
- Batch functions (based on Kubernetes jobs, {ref}`Spark <spark-operator>`, {ref}`Dask <dask-overview>`, {ref}`Horovod <horovod>`, etc.)

Function objects are all inclusive (code, spec, API, and metadata definitions), which allows placing them 
in a shared and versioned function market place. This means that different members of the team can produce or 
consume functions. Each function is versioned and stored in the MLRun database with a unique hash code, 
and gets a new hash code upon changes.

**In this section**

```{toctree}
:maxdepth: 1

functions-architecture
../concepts/functions-overview
create-and-use-functions
mlrun_code_annotations
function-storage
images
image-build
../concepts/node-affinity
configuring-job-resources
load-from-hub
```