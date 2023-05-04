# MLRun Community Edition Deployer

CLI tool for deploying MLRun Community Edition.
The CLI supports 3 commands:
- `deploy`: Deploys (or upgrades) an MLRun Community Edition Stack.
- `delete`: Uninstalls the CE and cleans up dangling resources.
- `patch-minikube-images`: If using custom images and running from Minikube, this command will patch the images to the Minikube env.

## Command Usage:
```
$ python automation/deployment/ce.py --help       
Usage: ce.py [OPTIONS] COMMAND [ARGS]...

  MLRun Community Edition Deployment CLI Tool

Options:
  --help  Show this message and exit.

Commands:
  delete                 Uninstall MLRun Community Edition Deployment
  deploy                 Deploy MLRun Community Edition
  patch-minikube-images  Patch MLRun Community Edition Deployment images...
```

### Deploy:
```
$ python automation/deployment/ce.py deploy --help               
Usage: ce.py deploy [OPTIONS]

  Deploy (or upgrade) MLRun Community Edition

Options:
  -cv, --chart-version TEXT       Version of the mlrun chart to install. If
                                  not specified, will install the latest
                                  version
  --devel                         Get the latest RC version of the mlrun
                                  chart. (Only works if --chart-version is not
                                  specified)
  --disable-pipelines             Disable the installation of Kubeflow
                                  Pipelines
  --disable-prometheus-stack      Disable the installation of the Prometheus
                                  stack
  --disable-spark-operator        Disable the installation of the Spark
                                  operator
  -f, --log-file TEXT             Path to log file. If not specified, will log
                                  only to stdout
  -m, --minikube                  Install the mlrun chart in local minikube.
  -mv, --mlrun-version TEXT       Version of mlrun to install. If not
                                  specified, will install the latest version
  -n, --namespace TEXT            Namespace to install the platform in.
                                  Defaults to 'mlrun'
  --override-jupyter-image TEXT   Override the jupyter image. Format:
                                  <repo>:<tag>
  --override-mlrun-api-image TEXT
                                  Override the mlrun-api image. Format:
                                  <repo>:<tag>
  --override-mlrun-ui-image TEXT  Override the mlrun-ui image. Format:
                                  <repo>:<tag>
  --registry-password TEXT        Password of the container registry to use
                                  for storing images
  --registry-secret-name TEXT     Name of the secret containing the
                                  credentials for the container registry to
                                  use for storing images
  --registry-url TEXT             URL of the container registry to use for
                                  storing images  [required]
  --registry-username TEXT        Username of the container registry to use
                                  for storing images
  --set TEXT                      Set custom values for the mlrun chart.
                                  Format: <key>=<value>
  --sqlite TEXT                   Path to sqlite file to use as the mlrun
                                  database. If not supplied, will use MySQL
                                  deployment
  --upgrade                       Upgrade the existing mlrun installation.
  -v, --verbose                   Enable debug logging
  --help                          Show this message and exit.
```

### Delete:
```
$ python automation/deployment/ce.py delete --help
Usage: ce.py delete [OPTIONS]

  Uninstall MLRun Community Edition Deployment

Options:
  --cleanup-namespace             Delete the namespace created during
                                  installation. This overrides the other
                                  cleanup options. WARNING: This will result
                                  in data loss!
  --cleanup-volumes               Delete the PVCs created during installation.
                                  WARNING: This will result in data loss!
  -f, --log-file TEXT             Path to log file. If not specified, will log
                                  only to stdout
  -n, --namespace TEXT            Namespace to install the platform in.
                                  Defaults to 'mlrun'
  --registry-secret-name TEXT     Name of the secret containing the
                                  credentials for the container registry to
                                  use for storing images
  --skip-cleanup-registry-secret  Skip deleting the registry secret created
                                  during installation
  --skip-uninstall                Skip uninstalling the Helm chart. Useful if
                                  already uninstalled and you want to perform
                                  cleanup only
  --sqlite TEXT                   Path to sqlite file to use as the mlrun
                                  database. If not supplied, will use MySQL
                                  deployment
  -v, --verbose                   Enable debug logging
  --help                          Show this message and exit.
```

### Patch Minikube Images:
```
$ python automation/deployment/ce.py patch-minikube-images --help
Usage: ce.py patch-minikube-images [OPTIONS]

  Patch MLRun Community Edition Deployment images to minikube. Useful if
  overriding images and running in minikube

Options:
  --jupyter-image TEXT    Override the jupyter image. Format: <repo>:<tag>
  -f, --log-file TEXT     Path to log file. If not specified, will log only to
                          stdout
  --mlrun-api-image TEXT  Override the mlrun-api image. Format: <repo>:<tag>
  --mlrun-ui-image TEXT   Override the mlrun-ui image. Format: <repo>:<tag>
  -v, --verbose           Enable debug logging
  --help                  Show this message and exit.
```