# dockerfiles and building

## build
to build run this command from the root directory of the mlrun repository:<br>

    MLRUN_DOCKER_TAG=X MLRUN_DOCKER_REPO=X MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX=X MLRUN_PYTHON_VERSION=X make push-docker-images

where:<br>
* `MLRUN_DOCKER_TAG` this is the tag created and pushed (like `latest` or `0.4.5`, defaults to `unstable`)
* `MLRUN_DOCKER_REPO` is your docker hub account (defaults to `mlrun`)
* `MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX` is some prefix common to the machine-learning/AI images created here (defaults to `ml-`)
* `MLRUN_PYTHON_VERSION` is the python version used (defaults to `3.7`).
* `MLRUN_MLUTILS_GITHUB_TAG` is the tag of the mlutils package installed in the images (defaults to `development`)


for example,
  `MLRUN_DOCKER_TAG=0.4.6 MLRUN_DOCKER_REPO=mlrun MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX=ml- MLRUN_PYTHON_VERSION=3.7 make docker-images`
this will generate the following images:
  * `mlrun/ml-base:0.4.6`       (python 3.7)
  * `mlrun/ml-models:0.4.6`     (python 3.7)
  * `mlrun/ml-models-gpu:0.4.6` (python 3.7) 
  * `mlrun/mlrun-api:0.4.6`     (python 3.7)
  * `mlrun/mlrun:0.4.6`         (python 3.7)

For compatability with some packages requiring py36, there is also an `ml-xxx` series of
images tagged `0.4.6-py36`

## notable changes
* `mlrun/dask` has been deprecated, use `mlrun/ml-base` or `mlrun/ml-models` instead
* `ml-models` and `ml-models-gpu` both contain OpenMPI and Horovod (and Dask)
* `plotly` has been added to `ml-models` and `ml-models-gpu`, see **[plotly python](https://plotly.com/python/)** for details

To run an image locally and explore its contents:  **`docker run -it mlrun/XXXXXX:0.4.7 /bin/bash`**<br>
or to load python (or run a script): **`docker run -it mlrun/XXXXXX:0.4.7 python`**.  
