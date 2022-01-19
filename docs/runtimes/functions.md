(Functions)=
# Functions & Job Submission

All the executions in MLRun are based on **Serverless Functions**, the functions allow specifying code and 
all the operational aspects (image, required packages, cpu/mem/gpu resources, storage, environment, etc.), 
the [different function runtimes](Function_runtimes) take care of automatically transforming the code and spec to fully 
managed and elastic services over Kubernetes which save significant operational overhead, 
address scalability and reduce infrastructure costs.

MLRun supports:
- Real-time functions for: serving, APIs, and stream processing (based on the high-performance Nuclio engine).
- Batch functions (based on Kubernetes jobs, Spark, Dask, Horovod, etc.) 

Function objects are all inclusive (code +  spec + API and metadata definitions) which allow placing them 
in a shared and versioned function market place, this way different members of the team can produce or 
consume functions. Each function is versioned and stored in the MLRun database with a unique hash code, 
and gets a new hash code upon changes.
There is also an open [public marketplace](https://www.mlrun.org/marketplace/functions/) which store many pre-developed functions for
use in your projects. 

<img src="../_static/images/mlrun-functions.png" alt="mlrun-architecture" width="600"/><br>

Functions can scale-out across multiple containers. Read more in [**Distributed and Parallel Jobs**](./distributed.md).

**MLRun batch functions and tasks**

Batch functions accept a **Task** (parameters, inputs, secrets, etc.) and return a **{py:class}`~mlrun.model.RunObject`** 
that hosts the status, results, data outputs, logs, etc. Every execution has a unique Run ID used for tracking.
Tasks can be broken to multiple child tasks (called `Iterations`), so that you can run a sequence of 
hyper-parameter or AutoML jobs. 

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
