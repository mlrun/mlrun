(Function_runtimes)=
# Function Runtimes

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
