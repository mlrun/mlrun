# dockerfiles and build script


to build run **`docker-build REPO PREFIX MLRUN_TAG NEW_TAG`** where:
* `REPO` is your docker hub account (like `mlrun`)
* `PREFIX` is some prefix common to all the images created here (like `ml`)
* `MLRUN_TAG` is a specific mlrun commit or tag _(prefix only tags with a `v`, like `v0.4.5`)_
* `NEW_TAG` in addition to MLRUN_TAG, a second tag is created with this name (like `latest` or `0.4.5`)

the script generates 2 sets of 4 images:
* one set with the current commit number
* one set with another user-defined name, say `latest`, or `issues-015`...

if your docker hub user is **`mlrun`** and the prefix is **`ml`** the 4 images will
be named:
  * **`mlrun/ml-base`**
    - base image for file transfers and other simple functions / jobs
    - includes `mlrun` and `kfp`
    - base image for all other images here
    - (WIP) this is a bloated image that should be minimal, remove conda and revert to minimal python with arrow
  * **`mlrun/ml-models`**
    - an almost complete set of AI tools, based on intel's python conda distribution
    - tensorflow > 2
    - xgboost, lightgbm
    - daal4py
    - other popular ml/ai packages
  * **`mlrun/ml-dask`**
    - can be used to launch 'dask' runtime jobs
    - adds a **`dask`** layer to **`ml-models`** 
    - complete distributed/kubernetes functionality
  * **`mlrun/ml-serving`**
    - derived from the base image, provides only predict/serving functionality
    - (WIP) what is minimal set of packages for this to predict/serve models

WIP:
* **`horovod-cpu`** and **`horovod-gpu`**
* openBLAS version of **`ml-models`** (no MKL)
* NVIDIA optimized base images
* ONNX serving...


To run an image locally and explore its contents:  **`docker run -it REPO/PREFIX-ml:0.4.5 /bin/bash`**<br>
or to load python (or run a script): **`docker run -it REPO/PREFIX-ml:0.4.5 python`**.  
