(images-usage)=
# Images and their usage in MLRun

Every release of MLRun includes several images for different usages. The build and the infrastructure images are described, and located, in the [README](https://github.com/mlrun/mlrun/blob/development/dockerfiles/README.md). They are also published to [dockerhub](https://hub.docker.com/u/mlrun) and [quay.io](https://quay.io/organization/mlrun).

**In this section**
- [Using images](#using-images)
- [MLRun images and how to build them](#mlrun-images-and-how-to-build-them)
- [MLRun images and external docker images](#mlrun-images-and-external-docker-images)

## Using images

See {ref}`build-function-image`.

## MLRun images and how to build them 

See [README](https://github.com/mlrun/mlrun/blob/development/dockerfiles/README.md).

## MLRun images and external docker images

There is no difference in the usage between the MLRun images and external docker images. However:
- MLRun images resolve auto tags: If you specify ```image="mlrun/mlrun"``` the API fills in the tag by the client version, e.g. changes it to `mlrun/mlrun:1.3.0`. So, if the client gets upgraded you'll automatically get a new image tag. 
- Where the data node registry exists, MLRun Appends the registry prefix, so the image loads from the datanode registry. This pulls the image more quickly, and also supports air-gapped sites. When you specify an MLRun image, for example `mlrun/mlrun:1.3.0`, the actual image used is similar to `datanode-registry.iguazio-platform.app.vm/mlrun/mlrun:1.3.0`.

These characteristics are great when you’re working in a POC or development environment. But MLRun typically upgrades packages as part of the image, and therefore the default MLRun images can break your product flow. 

### Working with images in production
```{admonition} Warning
For production, **create your own images** to ensure that the image is fixed.
```

- Pin the image tag, e.g. `image="mlrun/mlrun:1.3.0"`. This maintains the image tag at the version you specified, even when the client is upgraded. Otherwise, an upgrade of the client would also upgrade the image. (If you specify an external (not MLRun images) docker image, like python, the result is the docker/k8s default behavior, which defaults to `latest` when the tag is not provided.)
- Pin the versions of requirements, again to avoid breakages, e.g. `pandas==1.4.0`. (If you only specify the package name, e.g. pandas, then pip/conda (python's package managers) just pick up the latest version.)
