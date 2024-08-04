(images-usage)=
# Images and their usage in MLRun

Every release of MLRun includes several images for different usages. The build and the infrastructure images are described, and located, in the [README](https://github.com/mlrun/mlrun/blob/development/dockerfiles/README.md). They are also published to [DockerHub](https://hub.docker.com/u/mlrun) and [quay.io](https://quay.io/organization/mlrun).

This release of MLRun supports only Python 3.9.

**In this section**
- [Using images](#using-images)
- [MLRun images](#mlrun-images)
- [Building MLRun images](#building-mlrun-images)
- [Building a docker image using a Dockerfile and using it](#building-a-docker-image-using-a-dockerfile-and-using-it)
- [MLRun images and external docker images](#mlrun-images-and-external-docker-images)

## Using images

See {ref}`build-function-image`.

## MLRun runtime images  

All images are published to 
[DockerHub](https://hub.docker.com/u/mlrun) and [quay.io](https://quay.io/organization/mlrun).

The images are:

- `mlrun/mlrun`: An MLRun image includes preinstalled OpenMPI and other ML packages. Useful as a base image for simple jobs. 
- `mlrun/mlrun-gpu`: The same as `mlrun/mlrun` but for GPUs, including Open MPI. 
- `mlrun/ml-base`: Image for file acquisition, compression, Dask jobs, simple training jobs and other utilities.
- `mlrun/jupyter`: An image with Jupyter giving a playground to use MLRun in the open source. Built on top of jupyter/scipy-notebook, with the addition of MLRun and several demos and examples.

```{admonition} Note
When using the `mlrun` or `mlrun-gpu` image, use PyTorch versions up to and including than 2.0.1, but not higher. 
You can build your own images with newer CUDA for later releases of PyTorch.
```



## Building MLRun images

To build all images, run this command from the root directory of the mlrun repository:

`MLRUN_VERSION=X MLRUN_DOCKER_REPO=X MLRUN_DOCKER_REGISTRY=X make docker-images`

Where:
- MLRUN_VERSION is used as the tag of the image and also as the version injected into the code (e.g. latest or 0.7.0 or 0.6.5-rc6, defaults to unstable)
- MLRUN_DOCKER_REPO is the docker repository (defaults to mlrun)
- MLRUN_DOCKER_REGISTRY is the docker registry (e.g. quay.io/, gcr.io/, defaults to empty (docker hub))

For example, running `MLRUN_VERSION=x.y.z make docker-images` generates these images:
- mlrun/mlrun-api:x.y.z
- mlrun/mlrun:x.y.z
- mlrun/mlrun-gpu:x.y.z
- mlrun/jupyter:x.y.z
- mlrun/ml-base:x.y.z

You can also build only a specific image, for example, `make mlrun` (builds only the api image).

The possible commands are:
- `mlrun`
- `mlrun-gpu`

To run an image locally and explore its contents: `docker run -it <image-name>:<image-tag> /bin/bash`
or to load python (or run a script): `docker run -it <image-name>:<image-tag> python`

## Building a docker image using a Dockerfile and using it

This flow describes how to build the image externally, put it your private repo, and use it in MLRun.

1. Build an image using Dockerfile:
   1. Create a Dockerfile
   ```
   FROM mlrun/mlrun:X.X
   RUN pip install package1
   RUN pip install package2
   ```
   2. Build the image:
	```
	docker build -t your_docker_registry/your_image_name:tag
	```
   3. Push the image:
   ```
   docker push your_docker_registry/your_image_name:tag
   ```
2. Create a secret on K8s level for accessing your registry:
   ```
   kubectl --namespace default-tenant create secret docker-registry registry-credentials \
       --docker-server your-docker-registry \
       --docker-username <    > \
       --docker-password <    > \
       --docker-email <    >
   ```
3. In the code, use the image you created and provide the secret for pulling it:
   ```
   func = prj.set_function(name="func",...)
   func.set_image_pull_configuration(image_pull_secret_name="registry-credentials")
   ```
   Now when you run the function, the image is used.


## MLRun images and external docker images

There is no difference in the usage between the MLRun images and external docker images. However:
- MLRun images resolve auto tags: If you specify ```image="mlrun/mlrun"``` the API fills in the tag by the client version, e.g. changes it to `mlrun/mlrun:1.6.4`. So, if the client gets upgraded you'll automatically get a new image tag. 
- Where the data node registry exists, MLRun Appends the registry prefix, so the image loads from the data node registry. This pulls the image more quickly, and also supports air-gapped sites. When you specify an MLRun image, for example `mlrun/mlrun:1.6.4`, the actual image used is similar to `datanode-registry.iguazio-platform.app.vm/mlrun/mlrun:1.6.4`.

These characteristics are great when you’re working in a POC or development environment. But MLRun typically upgrades packages as part of the image, and therefore the default MLRun images can break your product flow. 

### Working with images in production
```{admonition} Warning
For production, **create your own images** to ensure that the image is fixed.
```

- Pin the image tag, e.g. `image="mlrun/mlrun:1.6.4"`. This maintains the image tag at the version you specified, even when the client is upgraded. Otherwise, an upgrade of the client would also upgrade the image. (If you specify an external (not MLRun images) docker image, like python, the result is the docker/k8s default behavior, which defaults to `latest` when the tag is not provided.)
- Pin the versions of requirements, again to avoid breakages, e.g. `pandas==1.4.0`. (If you only specify the package name, e.g. pandas, then pip/conda (python's package managers) just pick up the latest version.)
