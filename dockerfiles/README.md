# Dockerfiles and building

## build
to build all images run this command from the root directory of the mlrun repository:<br>

    MLRUN_VERSION=X MLRUN_DOCKER_REPO=X MLRUN_DOCKER_REGISTRY=X make docker-images

where:<br>
* `MLRUN_VERSION` this is used as the tag of the image and also as the version injected into the code (e.g. `latest` or `0.7.0` or `0.6.5-rc6`, defaults to `unstable`)
* `MLRUN_DOCKER_REPO` is the docker repository (defaults to `mlrun`)
* `MLRUN_DOCKER_REGISTRY` is the docker registry (e.g. `quay.io/`, `gcr.io/`, defaults to empty (docker hub))


for example, running `MLRUN_VERSION=0.7.0 make docker-images` will generate the following images:
  * `mlrun/mlrun-api:0.7.0`
  * `mlrun/mlrun:0.7.0`
  * `mlrun/jupyter:0.7.0`
  * `mlrun/ml-base:0.7.0`
  * `mlrun/ml-base:0.7.0-py36`
  * `mlrun/ml-models:0.7.0`
  * `mlrun/ml-models:0.7.0-py36`
  * `mlrun/ml-models-gpu:0.7.0` 
  * `mlrun/ml-models-gpu:0.7.0-py36`

(For compatibility with some packages requiring py36, there is also an `ml-xxx` series of
images tagged with the `-py36` suffix, e.g. `0.7.0-py36`)

It's also possible to build only a specific image - `make api` (will build only the api image)<br>
Or a set of images - `make mlrun jupyter base`
The possible commands are:
* `mlrun`
* `api`
* `jupyter`
* `base`
* `base-legacy`
* `models`
* `models-legacy`
* `models-gpu`
* `models-gpu-legacy`

To run an image locally and explore its contents:  **`docker run -it <image-name>:<image-tag> /bin/bash`**<br>
or to load python (or run a script): **`docker run -it <image-name>:<image-tag> python`**.  
