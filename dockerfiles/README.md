# MLRun Images
## Info
Every release of MLRun includes several images for different usages.
All images are published to [DockerHub](https://hub.docker.com/u/mlrun) and [quay.io](https://quay.io/organization/mlrun).

The images are:
* `mlrun/mlrun` - An MLRun image includes preinstalled OpenMPI and other ML packages. Image for file acquisition, compression, Dask jobs, simple training jobs and other utilities.
* `mlrun/mlrun-gpu` - Same as `mlrun/mlrun` but for GPUs, including `OPMI` (Available for MLRun >= 1.5.0)
* `mlrun/jupyter` - An image with [Jupyter](https://jupyter.org/) giving a playground to use MLRun in the open source.
  Built on top of [`jupyter/scipy-notebook`](
  https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-scipy-notebook), with the addition
  of MLRun and several demos and examples
* `mlrun/mlrun-api` - The image used for running the MLRun API
* `mlrun/mlrun-ui` - The image used for running the MLRun UI

**Deprecated images:** 

Image removed in MLRun 1.10.0:
* `mlrun/ml-base` - Image for file acquisition, compression, Dask jobs, simple training jobs and other utilities. In v1.10.0 replaced by `mlrun/mlrun`.

Image removed in MLRun 1.5.0:
* `mlrun/ml-models` - Image for analyzing data, model training and deep learning on CPUs. Built on top of 
  `mlrun/ml-base` with the addition of [Open MPI](https://www.open-mpi.org/), [PyTorch](https://pytorch.org/), 
  [TensorFlow](https://www.tensorflow.org/), [Horovod](https://horovod.ai/) and other [python packages](
  ./models/requirements.txt)


## Build
To build all images run this command from the root directory of the mlrun repository:<br>

    MLRUN_VERSION=X MLRUN_DOCKER_REPO=X MLRUN_DOCKER_REGISTRY=X make docker-images

Where:<br>
* `MLRUN_VERSION` this is used as the tag of the image and also as the version injected into the code (e.g. `latest` or `0.7.0` or `0.6.5-rc6`, defaults to `unstable`)
* `MLRUN_DOCKER_REPO` is the docker repository (defaults to `mlrun`)
* `MLRUN_DOCKER_REGISTRY` is the docker registry (e.g. `quay.io/`, `gcr.io/`, defaults to empty (docker hub))


For example, running `MLRUN_VERSION=x.y.z make docker-images` will generate the following images:
  * `mlrun/mlrun-api:x.y.z`
  * `mlrun/mlrun:x.y.z`
  * `mlrun/mlrun-gpu:x.y.z`
  * `mlrun/jupyter:x.y.z`

It's also possible to build only a specific image - `make api` (will build only the api image)<br>
Or a set of images - `make mlrun jupyter base`
The possible commands are:
* `mlrun`
* `mlrun-gpu`
* `api`
* `jupyter`
* `base`

To run an image locally and explore its contents:  `docker run -it <image-name>:<image-tag> /bin/bash`<br>
or to load python (or run a script): `docker run -it <image-name>:<image-tag> python`.

## Test image flavors

The Dockerized test image (`dockerfiles/test/Dockerfile`) is built in three flavors, selected by the
`MLRUN_TEST_FLAVOR` build-arg (also exposed as a make variable):

* `all` (default) — the combined suite. Installs both the full KFP 1.8 stack
  (`mlrun-pipelines-kfp-v1-8[kfp]`) and the api-server requirements. Reproduces the pre-split test
  image and is used by every dockerized-test job except the split unit-test matrix (integration,
  migrations, backward-compatibility, docs).
* `client` — the rest/SDK unit-test suite. Installs the full KFP 1.8 stack and **excludes** the
  api-server requirements.
* `server` — the api unit-test suite. Installs the api-server requirements and the KFP adapter
  **without** the `[kfp]` extra (`kfp-server-api` only).

Each flavor has its own locked-requirements file
(`dockerfiles/test/locked-requirements-<flavor>.txt`), regenerated via
`make upgrade-mlrun-test-all-deps-lock` / `-client-` / `-server-` (all covered by the aggregate
`make upgrade-mlrun-deps-lock`). The flavor is appended to the image and cache tags so the flavors
never collide.

A plain `make build-test` / `make test-dockerized` builds the **all** flavor; pass
`MLRUN_TEST_FLAVOR=client` or `=server` to build or run a specific suite, e.g.:

    MLRUN_TEST_FLAVOR=server make test-dockerized
