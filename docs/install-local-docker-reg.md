# Install MLRun on a local Docker registry

To use MLRun with your local Docker registry, run the MLRun API service, dashboard, and example Jupyter server, by using the following script.

```{admonition} Notes
- The docker deployment doesn't include the real time function (Nuclio) framework.
- Using Docker is limited to local runtimes. (Remote works only on kuberenetes.)
- By default, the MLRun API service runs inside the Jupyter server. Set the MLRUN_DBPATH env var in Jupyter to point to an alternative service address.
- The artifacts and DB are stored under **/home/jovyan/data**. Use the docker -v option to persist the content on the host (e.g. `-v $(SHARED_DIR}:/home/jovyan/data`).
- If Docker is running on Windows with WSL 2, you must create a SHARED_DIR before running these commands. Provide the full path when executing  (e.g. `mkdir /mnt/c/mlrun-data`  `SHARED_DIR=/mnt/c/mlrun-data`).
```

```sh
SHARED_DIR=~/mlrun-data

docker pull mlrun/jupyter:1.0.0
docker pull mlrun/mlrun-ui:1.0.0

docker network create mlrun-network
docker run -it -p 8080:8080 -p 30040:8888 --rm -d --network mlrun-network --name jupyter -v ${SHARED_DIR}:/home/jovyan/data mlrun/jupyter:1.0.0
docker run -it -p 30050:80 --rm -d --network mlrun-network --name mlrun-ui -e MLRUN_API_PROXY_URL=http://jupyter:8080 mlrun/mlrun-ui:1.0.0
```

When the execution completes:

- Open Jupyter Lab on port 30040 and run the code in the [**mlrun_basics.ipynb**](https://github.com/mlrun/mlrun/blob/master/examples/mlrun_basics.ipynb) notebook.
- Use the MLRun dashboard on port 30050.
