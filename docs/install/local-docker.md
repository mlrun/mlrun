(install-local-docker)=
# Install MLRun locally using Docker

You can install and use MLRun and Nuclio locally on your computer. This does not include all the services and elastic 
scaling capabilities, which you can get with the Kubernetes based deployment, but it is much simpler to start with.

```{admonition} Note
Using Docker is limited to local, Nuclio, and Serving runtimes and local pipelines.
```

Use [`docker compose`](https://docs.docker.com/compose/) to install MLRun. It deploys the MLRun service,
MLRun UI, Nuclio serverless engine, and optionally the Jupyter server.

There are two installation options:
- [**Use MLRun with your own client (PyCharm, VSCode, Jupyter)**](#use-mlrun-with-your-own-client)
- [**Use MLRun with MLRun Jupyter image (pre loaded with examples/demos)**](#use-mlrun-with-mlrun-jupyter-image)

In both cases you need to set the `SHARED_DIR` environment variable to point to a host path for storing MLRun artifacts and DB, 
for example `export SHARED_DIR=~/mlrun-data` (or use `set SHARED_DIR=c:\mlrun-data` in windows), make sure the directory exists.

It is recommended to set the `HOST_IP` variable with your computer IP address (required for Nuclio dashboard). 
You can select a specific MLRun version with the `TAG` variable and Nuclio version with the `NUCLIO_TAG` variable.

Add the `-d` flag to `docker-compose` for running in detached mode (in the background).

```{admonition} Note
We added support for running as a non-root user in 1.0.5, hence the underlaying exposed port was changed.
If you want to use previous mlrun versions, modify the mlrun-ui port from 8090 back to 80.
```

## Use MLRun with your own client

The following commands install MLRun + Nuclio for work with your own IDE or notebook. 

**{Download}`[Download here]<./compose.yaml>`** the `compose.yaml` file, save it to the working dir and type:
````{dropdown} show the compose.yaml file
   ```{literalinclude} ./compose.yaml
   :language: yaml
   ```
````

`````{tab-set}
````{tab-item} Linux/Mac
```sh
export HOST_IP=<your host IP address>
export SHARED_DIR=~/mlrun-data
mkdir $SHARED_DIR -p
docker-compose -f compose.yaml up
``` 

Your `HOST_IP` address can be found using the `ip addr` or `ifconfig` commands. It is recomended to select an address that does not change dynamically (for example the IP of the bridge interface).
````

````{tab-item} Windows (cmd)
```sh
set HOST_IP=<your host IP address>
set SHARED_DIR=c:\mlrun-data
mkdir %SHARED_DIR%
docker-compose -f compose.yaml up
``` 

Your `HOST_IP` address can be found using the `ipconfig` shell command, it is recomended to select an address which does not change dynamically (for example the IP of the `vEthernet` interface).
````
`````

This creates 3 services:
- MLRun API (in [http://localhost:8080](http://localhost:8080))
- MLRun UI (in [http://localhost:8060](http://localhost:8060))
- Nuclio Dashboard/controller (in [http://localhost:8070](http://localhost:8070))

After installing MLRun service, set your client environment to work with the service, by setting the MLRun path env variable to 
`MLRUN_DBPATH=http://localhost:8080` or using `.env` files (see [setting client environment](./remote.html)).

## Use MLRun with MLRun Jupyter image

For the quickest experience with MLRun you can deploy MLRun with a pre integrated Jupyter server loaded with various ready-to-use MLRun examples.

**{Download}`[Download here]<./compose.with-jupyter.yaml>`** the `compose.with-jupyter.yaml` file, save it to the working dir and type:
````{toggle} show the compose.with-jupyter.yaml file
   ```{literalinclude} ./compose.with-jupyter.yaml
   :language: yaml
   ```
````

`````{tab-set}
````{tab-item} Linux/Mac
```sh
export HOST_IP=<your host IP address>
export SHARED_DIR=~/mlrun-data
mkdir $SHARED_DIR -p
docker-compose -f compose.with-jupyter.yaml up
```

Your `HOST_IP` address can be found using the `ip addr` or `ifconfig` commands. It is recomended to select an address that does not change dynamically (for example the IP of the bridge interface). 
````

````{tab-item} Windows (cmd)
```sh
set HOST_IP=<your host IP address>
set SHARED_DIR=c:\mlrun-data
mkdir %SHARED_DIR%
docker-compose -f compose.with-jupyter.yaml up
``` 

Your `HOST_IP` address can be found using the `ipconfig` shell command, it is recomended to select an address which does not change dynamically (for example the IP of the `vEthernet` interface).
````
`````

This creates 4 services:
- Jupyter lab (in [http://localhost:8888](http://localhost:8888))
- MLRun API (in [http://localhost:8080](http://localhost:8080)), running on the Jupyter container
- MLRun UI (in [http://localhost:8060](http://localhost:8060))
- Nuclio Dashboard/controller (in [http://localhost:8070](http://localhost:8070))

After the installation, access the Jupyter server (in [http://localhost:8888](http://localhost:8888)) and run through the [`quick-start` tutorial](../quick-start/quick-start.html) and `demos`.
You can see the projects, tasks, and artifacts in MLRun UI (in [http://localhost:8060](http://localhost:8060))

The Jupyter environment is pre-configured to work with the local MLRun and Nuclio services. 
You can switch to a remote or managed MLRun cluster by editing the `mlrun.env` file in the Jupyter files tree.

The artifacts and DB are stored under **/home/jovyan/data** (`/data` in Jupyter tree). 
