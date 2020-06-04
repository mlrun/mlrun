# dockerfiles and building

## build

to build run this command from the root directory of the mlrun repository:  

    MLRUN_DOCKER_TAG=X MLRUN_DOCKER_REPO=X MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX=X MLRUN_PACKAGE_TAG=X MLRUN_GITHUB_REPO=X MLRUN_PYTHON_VERSION=X make push-docker-images

where:  

* `MLRUN_DOCKER_TAG` this is the tag created and pushed (like `latest` or `0.4.7`, defaults to `latest`)
* `MLRUN_DOCKER_REPO` is your docker hub account (defaults to `mlrun`)
* `MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX` is some prefix common to the machine-learning/AI images created here (defaults to `ml-`)
* `MLRUN_PACKAGE_TAG` is the tag of mlrun package installed in the images _(defaults to `development`, prefix tags with a `v`, like `v0.4.5`, or use the commit id SHA)_
* `MLRUN_GITHUB_REPO` is the github repo from which we `pip install` mlrun (defaults to `mlrun`)
* `MLRUN_PYTHON_VERSION` is the python version used (defaults to `3.7`).

for example,
  `MLRUN_DOCKER_TAG=0.4.7 MLRUN_DOCKER_REPO=mlrun MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX=ml- MLRUN_PACKAGE_TAG=v0.4.7 MLRUN_GITHUB_REPO=mlrun MLRUN_PYTHON_VERSION=3.7 make docker-images`
this will generate the following images:

* `mlrun/ml-base:0.4.7`       (python 3.7)
* `mlrun/ml-models:0.4.7`     (python 3.7)
* `mlrun/ml-models-gpu:0.4.7` (python 3.7)
* `mlrun/mlrun-api:0.4.7`     (python 3.7)
* `mlrun/mlrun:0.4.7`         (python 3.7)

For compatability with some packages requiring py36, there is also an `ml-xxx` series of
images tagged `0.4.7-py36`

## notable changes

* `ml-models` and `ml-models-gpu` both contain OpenMPI and Horovod (and Dask)
* `plotly` has been added to `ml-models` and `ml-models-gpu`, see **[plotly python](https://plotly.com/python/)** for details

To run an image locally and explore its contents:  **`docker run -it mlrun/XXXXXX:0.4.7 /bin/bash`**  
or to load python (or run a script): **`docker run -it mlrun/XXXXXX:0.4.7 python`**.  
