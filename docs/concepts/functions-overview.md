(Functions-overview)=
# Overview

All the executions in MLRun are based on **Serverless Functions**. The functions allow specifying code and 
all the operational aspects (image, required packages, cpu/mem/gpu resources, storage, environment, etc.). 
The [different function runtimes](Function_runtimes) take care of automatically transforming the code and spec to fully 
managed and elastic services over Kubernetes, which saves significant operational overhead, 
addresses scalability and reduces infrastructure costs.

MLRun supports:
- Real-time functions for: serving, APIs, and stream processing (based on the high-performance Nuclio engine). 
- Batch functions: based on Kubernetes jobs, Spark, Dask, Horovod, etc.

Function objects are all inclusive (code, spec, API, and metadata definitions), which allows placing them 
in a shared and versioned function market place. This means that different members of the team can produce or 
consume functions. Each function is versioned and stored in the MLRun database with a unique hash code, 
and gets a new hash code upon changes.

MLRun also has an open [public marketplace](https://www.mlrun.org/marketplace/functions/) that stores many pre-developed functions for
use in your projects. 

<img src="../_static/images/mlrun-functions.png" alt="mlrun-architecture" width="600"/><br>


(Function_runtimes)=
## Function Runtimes

When you create an MLRun function you need to specify a runtime kind (e.g. `kind='job'`). Each runtime supports 
its own specific attributes (e.g. Jars for Spark, Triggers for Nuclio, Auto-scaling for Dask, etc.).

MLRun supports these runtimes:

Real-time runtimes:
* **nuclio** - real-time serverless functions over Nuclio
* **serving** - higher level real-time Graph (DAG) over one or more Nuclio functions

Batch runtimes:
* **handler** - execute python handler (used automatically in notebooks or for debug)
* **local** - execute a Python or shell program 
* **job** - run the code in a Kubernetes Pod
* **dask** - run the code as a Dask Distributed job (over Kubernetes)
* **mpijob** - run distributed jobs and Horovod over the MPI job operator, used mainly for deep learning jobs 
* **spark** - run the job as a Spark job (using Spark Kubernetes Operator)
* **remote-spark** - run the job on a remote Spark service/cluster (e.g. Iguazio Spark service)

**Common attributes for Kubernetes based functions** 

All the Kubernetes based runtimes (Job, Dask, Spark, Nuclio, MPIJob, Serving) support a common 
set of spec attributes and methods for setting the PODs:

function.spec attributes (similar to k8s pod spec attributes):
* volumes
* volume_mounts
* env
* resources
* replicas
* image_pull_policy
* service_account
* image_pull_secret

common function methods:
* set_env(name, value)
* set_envs(env_vars)
* gpus(gpus, gpu_type)
* with_limits(mem, cpu, gpus, gpu_type)
* with_requests(mem, cpu)
* set_env_from_secret(name, secret, secret_key)
