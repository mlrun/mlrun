# Images and their usage in MLRun

## Supported images
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

Note: For compatibility with some packages requiring py36, there is also a tag with the `-py36` suffix (e.g. 
`0.7.0-py36`) for the ml images (`mlrun/ml-base`, `mlrun/ml-models`, `mlrun/ml-models-gpu`).

## Building images
To build all images, run this command from the root directory of the mlrun repository:<br>

    MLRUN_VERSION=X MLRUN_DOCKER_REPO=X MLRUN_DOCKER_REGISTRY=X make docker-images

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

## MLRun images vs. external docker images

There is no difference in the usage between the MLRun images and external docker images. However, MLRun images:
- Resolve auto tags: If you specify `image="mlrun/mlrun"` the API fills in the tag by its version, e.g. changes it to `mlrun/mlrun:0.9.2`. 
And if the API gets upgraded you'll automatically get a new API image. 
- Append the registry prefix, saving the image in the datanode registry, except for any third party k8s installations. This pulls the image more 
quickly, and also supports air-gapped sites. When you specify an MLRun image, for example `mlrun/mlrun:0.9.1` the actual image used is similar to 
datanode-registry.iguazio-platform.app.vm.

These characteristics are great when you’re working in a POC or development environment. But MLRun typically upgrades packages as part of the image, and therefore 
the default MLRun images can break your product flow. 

For production you should create your own images:
- Pin the image tag, e.g. `image="mlrun/mlrun:0.10.0"`. This maintains the image tag at 0.10.0 even when the API is upgraded. Otherwise, an upgrade of the API 
would also upgrade the image. (If you specify `mlrun/mlrun` the result is the docker/k8s default behavior, which defaults to `latest` when the tag is not provided. 
By providing tags, you ensure that the image is fixed.)
- Specify the exact packages (e.g. specific tensorflow or pytorch package) your system uses.
- Pin the versions of requirements, again to avoid breakages, e.g. pandas==1.4.0. (If you only specify the package name, e.g. pandas, 
then pip/conda (python's package managers) just pick up the latest version.)

## Using images

See [Kubernetes Jobs & Images](./runtimes/mlrun_jobs.ipynb)