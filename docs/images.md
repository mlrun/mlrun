# Images and their usage in MLRun

Every release of MLRun includes several images for different usages. The build and the infrastructure images are described, and located, in the [README](https://github.com/mlrun/mlrun/blob/development/dockerfiles/README.md). They are also published to [dockerhub](https://hub.docker.com/u/mlrun) and [quay.io](https://quay.io/organization/mlrun).

## Using images

See [Kubernetes Jobs & Images](./runtimes/mlrun_jobs.ipynb)

## MLRun images and how to build them 

See [README](https://github.com/mlrun/mlrun/blob/development/dockerfiles/README.md).

## MLRun images and external docker images

There is no difference in the usage between the MLRun images and external docker images. However, MLRun images:
- Resolve auto tags: If you specify ```image="mlrun/mlrun"``` the API fills in the tag by its version, e.g. changes it to `mlrun/mlrun:0.9.2`. And if the API gets upgraded you'll automatically get a new API image. 
- Append the registry prefix, loading the image from the datanode registry, except for any third party k8s installations. This pulls the image more quickly, and also supports air-gapped sites. When you specify an MLRun image, for example `mlrun/mlrun:0.9.2`, the actual image used is similar to `datanode-registry.iguazio-platform.app.vm`.

These characteristics are great when you’re working in a POC or development environment. But MLRun typically upgrades packages as part of the image, and therefore the default MLRun images can break your product flow. 

### Working with images in production
For production you should create your own images to ensure that the image is fixed.
- Pin the image tag, e.g. `image="mlrun/mlrun:0.10.0"`. This maintains the image tag at 0.10.0 even when the API is upgraded. Otherwise, an upgrade of the API would also upgrade the image. (If you specify `mlrun/mlrun` the result is the docker/k8s default behavior, which defaults to `latest` when the tag is not provided.)
- Specify the exact packages your application/use case requires (e.g. specific tensorflow or pytorch package).
- Pin the versions of requirements, again to avoid breakages, e.g. `pandas==1.4.0`. (If you only specify the package name, e.g. pandas, then pip/conda (python's package managers) just pick up the latest version.)