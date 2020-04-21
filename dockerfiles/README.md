# dockerfiles and build script

## build
to build run this script from the root directory of the mlrun repository:<br>

    ./docker-build REPO PREFIX MLRUN_TAG NEW_TAG PYTHON_VER_ML PYTHON_VER_CORE

where:<br>
* `REPO` is your docker hub account (like `mlrun`)
* `PREFIX` is some prefix common to the machine-learning/AI images created here (like `ml`)
* `MLRUN_TAG` is a specific mlrun commit or tag _(prefix only tags with a `v`, like `v0.4.5`)_
* `NEW_TAG` this is the tag created and pushed (like `latest` or `0.4.5`)
* `PYTHON_VER_ML` is the version for the ml-xxxx series. 
* `PYTHON_VER_CORE` is the python version for `httpd`, `dask` and `test`

for example,
  `./docker-build.sh mlrun ml v0.4.6 0.4.6 3.8 3.6`
this will generate the following images:
  * `mlrun/ml-base:0.4.6`       (python 3.8)
  * `mlrun/ml-models:0.4.6`     (python 3.8)
  * `mlrun/ml-models-gpu:0.4.6` (python 3.8) 
  * `mlrun/ml-serving:0.4.6`    (python 3.8)
  * `mlrun/mlrun-api:0.4.6`     (python 3.6)
  * `mlrun/dask:0.4.6`          (python 3.6)
  * `mlrun/mlrun:0.4.6`         (python 3.6)

## notable changes
* `ml-models` and `ml-models-gpu` both contain OpenMPI and Horovod
* `ml-hvd` and `ml-hvd-gpu` will be deprecated once testing is complete

To run an image locally and explore its contents:  **`docker run -it mlrun/XXXXXX:0.4.6 /bin/bash`**<br>
or to load python (or run a script): **`docker run -it mlrun/XXXXXX:0.4.5 python`**.  
