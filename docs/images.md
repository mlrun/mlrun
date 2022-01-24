# Images and their usage in MLRun

Every release of MLRun includes several images for different usages.
All images are published to [dockerhub](https://hub.docker.com/u/mlrun) and [quay.io](https://quay.io/organization/mlrun).

## Images for use in the jobs' pods

* `mlrun/mlrun`: The most basic (and smallest) image, can be used for simple jobs. Basically just MLRun installed on 
  top of a python image.
* `mlrun/ml-base`: Image for file acquisition, compression, dask jobs, simple training jobs and other utilities. Like
`mlrun/mlrun` with the addition of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and other [python 
  packages](./base/requirements.txt).
* `mlrun/ml-models`: Image for analyzing data, model training and deep learning on CPUs. Built on top of 
  `mlrun/ml-base` with the addition of [Open MPI](https://www.open-mpi.org/), [PyTorch](https://pytorch.org/), 
  [TensorFlow](https://www.tensorflow.org/), [Horovod](https://horovod.ai/) and other [python packages](
  ./models/requirements.txt).
* `mlrun/ml-models-gpu`: Same as `mlrun/ml-models` but for GPUs.

The files are located in [MLRun Dockerfiles](https://github.com/mlrun/mlrun/tree/development/dockerfiles).

## MLRun infrastructure images

See [README](https://github.com/mlrun/mlrun/blob/development/dockerfiles/README.md).

## Using images

See [Kubernetes Jobs & Images](./runtimes/mlrun_jobs.ipynb)

## Building images
To build all images, run this command from the root directory of the mlrun repository:<br>

    `MLRUN_VERSION=X MLRUN_DOCKER_REPO=X MLRUN_DOCKER_REGISTRY=X make docker-images`

Where:<br>
* `MLRUN_VERSION` is used as the tag of the image and is also the version injected into the code (e.g. `latest` or `0.9.0` or `0.9.2-rc1`, defaults to `unstable`)
* `MLRUN_DOCKER_REPO` is the docker repository (defaults to `mlrun`)
* `MLRUN_DOCKER_REGISTRY` is the docker registry (e.g. `quay.io/`, `gcr.io/`, defaults to empty (docker hub))


For example, running `MLRUN_VERSION=0.9.2 make docker-images` generates the following images:
  * `mlrun/mlrun-api:0.9.2`
  * `mlrun/mlrun:0.9.2`
  * `mlrun/jupyter:0.9.2`
  * `mlrun/ml-base:0.9.2`
  * `mlrun/ml-base:0.9.2-py36`
  * `mlrun/ml-models:0.9.2`
  * `mlrun/ml-models:0.9.2-py36`
  * `mlrun/ml-models-gpu:0.9.2` 
  * `mlrun/ml-models-gpu:0.9.2-py36`

You can build specific images. For example:
- To build only the api image: `make api`
- To build a set of images: `make mlrun jupyter base`

The supported commands are:
* `mlrun`
* `api`
* `jupyter`
* `base`
* `base-legacy`
* `models`
* `models-legacy`
* `models-gpu`
* `models-gpu-legacy`

To run an image locally and explore its contents: `docker run -it <image-name>:<image-tag> /bin/bash`<br>
or to load python (or run a script): `docker run -it <image-name>:<image-tag> python`.
