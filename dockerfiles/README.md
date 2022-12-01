# MLRun Images
## Info
Every release of MLRun includes several images for different usages.
All images are published to [dockerhub](https://hub.docker.com/u/mlrun) and [quay.io](https://quay.io/organization/mlrun).

The images are:
* `mlrun/mlrun` - The most basic (and smallest) image, can be used for simple jobs. Basically just MLRun installed on 
  top of a python image
* `mlrun/ml-base` - Image for file acquisition, compression, dask jobs, simple training jobs and other utilities. Like 
  `mlrun/mlrun` with the addition of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and other [python 
  packages](./base/requirements.txt) 
* `mlrun/ml-models` - Image for analyzing data, model training and deep learning on CPUs. Built on top of 
  `mlrun/ml-base` with the addition of [Open MPI](https://www.open-mpi.org/), [PyTorch](https://pytorch.org/), 
  [TensorFlow](https://www.tensorflow.org/), [Horovod](https://horovod.ai/) and other [python packages](
  ./models/requirements.txt)
* `mlrun/ml-models-gpu` - Same as `mlrun/ml-models` but for GPUs
* `mlrun/jupyter` - An image with [Jupyter](https://jupyter.org/) giving a playground to use MLRun in the open source.
  Built on top of [`jupyter/scipy-notebook`](
  https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-scipy-notebook), with the addition
  of MLRun and several demos and examples
* `mlrun/mlrun-api` - The image used for running the MLRun API
* `mlrun/mlrun-ui` - The image used for running the MLRun UI

## Build
To build all images run this command from the root directory of the mlrun repository:<br>

    MLRUN_VERSION=X MLRUN_DOCKER_REPO=X MLRUN_DOCKER_REGISTRY=X make docker-images

Where:<br>
* `MLRUN_VERSION` this is used as the tag of the image and also as the version injected into the code (e.g. `latest` or `0.7.0` or `0.6.5-rc6`, defaults to `unstable`)
* `MLRUN_DOCKER_REPO` is the docker repository (defaults to `mlrun`)
* `MLRUN_DOCKER_REGISTRY` is the docker registry (e.g. `quay.io/`, `gcr.io/`, defaults to empty (docker hub))


For example, running `MLRUN_VERSION=1.2.0 make docker-images` will generate the following images:
  * `mlrun/mlrun-api:1.2.0`
  * `mlrun/mlrun:1.2.0`
  * `mlrun/jupyter:1.2.0`
  * `mlrun/ml-base:1.2.0`
  * `mlrun/ml-models:1.2.0`
  * `mlrun/ml-models-gpu:1.2.0` 

It's also possible to build only a specific image - `make api` (will build only the api image)<br>
Or a set of images - `make mlrun jupyter base`
The possible commands are:
* `mlrun`
* `api`
* `jupyter`
* `base`
* `models`
* `models-gpu`

To run an image locally and explore its contents:  `docker run -it <image-name>:<image-tag> /bin/bash`<br>
or to load python (or run a script): `docker run -it <image-name>:<image-tag> python`.
