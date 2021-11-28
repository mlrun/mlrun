(Functions)=
# Functions & Job Submission

All the executions in MLRun are based on **Serverless Functions**, the functions allow specifying code and 
all the operational aspects (image, required packages, cpu/mem/gpu resources, storage, environment, etc.), 
the [different function runtimes](Function_runtimes) take care of automatically transforming the code and spec to fully 
managed and elastic services over Kubernetes which save significant operational overhead, 
address scalability and reduce infrastructure costs.

MLRun supports batch functions (based on Kubernetes jobs, Spark, Dask, Horovod, etc.) or Real-time functions 
for serving, APIs, and stream processing (based on the high-performance Nuclio engine).

Function objects are all inclusive (code +  spec + API and metadata definitions) which allow placing them 
in a shared and versioned function market place, this way different members of the team can produce or 
consume functions. Each function is versioned and stored in the MLRun database with a unique hash code, 
and gets a new hash code upon changes.
There is also an open [public marketplace](https://www.mlrun.org/marketplace/functions/) which store many pre-developed functions for
use in your projects. 

**MLRun Functions and Tasks**

Batch functions accepts a **Task** (parameters, inputs, secrets, etc.) and return a **{py:class}`~mlrun.model.RunObject`** 
which hosts the status, results, data outputs, logs, etc. every execution has a unique Run ID used for tracking.
Tasks can be broken to multiple child tasks (called `Iterations`), allowing to run a sequence of 
hyper-parameter or AutoML jobs. 


<img src="../_static/images/mlrun-functions.png" alt="mlrun-architecture" width="600"/><br>


Functions can scale-out across multiple containers, can read more about [**Distributed and Parallel Jobs**](./distributed.md)


**For more details and examples:**

```{toctree}
:maxdepth: 1
using-functions
function-storage
mlrun_code_annotations
code-archive
mlrun_jobs
node-affinity
load-from-marketplace
```
