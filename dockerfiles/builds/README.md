# builds

the dockerfiles contained in this folder will build packages that can then be copied to a local or remote filesystem using `docker create` and `docker cp`.

## openmpi

to build and copy out the tar file run the following

```python
docker build -f dockerfiles/builds/Dockerfile.openmpi \
             -t mlrun/openmpi-build \
             --build-arg OMPI=4.0.3 .

docker create -ti --name ompi mlrun/openmpi-build
docker cp ompi:/openmpi-4.0.3.tar.gz /tmp/
docker rm ompi
```

