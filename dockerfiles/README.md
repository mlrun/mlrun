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

The Dockerized test image (`dockerfiles/test/Dockerfile`) is built in two flavors, selected by the
`MLRUN_TEST_FLAVOR` build-arg (also exposed as a make variable):

* `client` (default) — the rest/SDK suite. Installs the full KFP 1.8 stack
  (`mlrun-pipelines-kfp-v1-8[kfp]`) and **excludes** the api-server requirements.
* `server` — the api suite. Installs the api-server requirements and the KFP adapter **without** the
  `[kfp]` extra (`kfp-server-api` only).

There is deliberately **no combined flavor**: every dockerized-test job runs on exactly one of these,
so the two locks can diverge independently (this is what lets the server suite later move to Pydantic 2
while the client suite stays on Pydantic 1 / KFP 1.8). Which job uses which flavor:

| Job (make target) | Flavor |
|---|---|
| unit tests — `test-dockerized` | `client` and `server` (CI matrix) |
| integration — `test-integration-dockerized` | `client` and `server` (CI matrix; the target scopes the pytest paths per flavor) |
| migrations — `test-migrations-dockerized` | `server` |
| backward-compat — `test-backward-compatibility-dockerized` | `server` |
| docs — `build-docs-dockerized` / `html-docs-dockerized` | `client` (default) |

Each flavor has its own locked-requirements file
(`dockerfiles/test/locked-requirements-<flavor>.txt`), regenerated via
`make upgrade-mlrun-test-client-deps-lock` / `-server-` (both covered by the aggregate
`make upgrade-mlrun-deps-lock`). The flavor is appended to the image and cache tags so the flavors
never collide.

A plain `make build-test` / `make test-dockerized` builds the **client** flavor; the server-side
targets self-select `server`. Pass `MLRUN_TEST_FLAVOR=server` explicitly to build or run the server
suite, e.g.:

    MLRUN_TEST_FLAVOR=server make test-dockerized
