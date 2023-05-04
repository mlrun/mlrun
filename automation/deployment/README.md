# MLRun Community Edition Deployer

CLI tool for deploying MLRun Community Edition.
The CLI supports 3 commands:
- `deploy`: Deploys (or upgrades) an MLRun Community Edition Stack.
- `delete`: Uninstalls the CE and cleans up dangling resources.
- `patch-minikube-images`: If using custom images and running from Minikube, this command will patch the images to the Minikube env.

## Command Usage:

### Deploy:
To deploy the CE the minimum needed is the registry url and registry credentials. You can run:
```
$ python automation/deployment/ce.py deploy \
    --registry-url <REGISTRY_URL> \
    --registry-username <REGISTRY_USERNAME> \
    --registry-password <REGISTRY_PASSWORD>
```
This will deploy the CE with the default configuration.

Instead of passing the registry credentials as command line arguments, you can create a secret in the cluster and pass the secret name:
```
$ python automation/deployment/ce.py deploy \
    --registry-url <REGISTRY_URL> \
    --registry-secret-name <REGISTRY_SECRET_NAME>
```

#### Extra Configurations:

You can override the mlrun version and chart version by using the flags `--mlrun-version` and `--chart-version` respectively.

To disable the components that are installed by default, you can use the following flags:
- `--disable-pipelines`: Disable the installation of Kubeflow Pipelines.
- `--disable-prometheus-stack`: Disable the installation of the Prometheus stack.
- `--disable-spark-operator`: Disable the installation of the Spark operator.

To override the images used by the CE, you can use the following flags:
- `--override-jupyter-image`: Override the jupyter image. Format: `<repo>:<tag>`
- `--override-mlrun-api-image`: Override the mlrun-api image. Format: `<repo>:<tag>`
- `--override-mlrun-ui-image`: Override the mlrun-ui image. Format: `<repo>:<tag>`

To run mlrun with sqlite instead of MySQL, you can use the `--sqlite` flag. The value should be the path to the sqlite file to use.

To set custom values for the mlrun chart, you can use the `--set` flag. The format is `<key>=<value>`. For example:
```
$ python automation/deployment/ce.py deploy \
    --registry-url <REGISTRY_URL> \
    --registry-username <REGISTRY_USERNAME> \
    --registry-password <REGISTRY_PASSWORD> \
    --set  mlrun.db.persistence.size="1Gi" \
    --set mlrun.api.persistence.size="1Gi"
```

To install the CE in a different namespace, you can use the `--namespace` flag.

To install the CE in minikube, you can use the `--minikube` flag.


### Upgrade
To upgrade the CE, you can use the same command as deploy with the flag `--upgrade`. 
The CLI will detect that the CE is already installed and will perform an upgrade. The flag will instruct helm to reuse values from the previous deployment.

### Delete:
To simply uninstall the CE deployment, you can run:
```
$ python automation/deployment/ce.py delete
```

To delete the CE deployment and clean up remaining volumes, you can run:
```
$ python automation/deployment/ce.py delete --cleanup-volumes
```

To cleanup the entire namespace, you can run:
```
$ python automation/deployment/ce.py delete --cleanup-namespace
```

If you already uninstalled, and only want to run cleanup, you can use the `--skip-uninstall` flag.


### Patch Minikube Images:
Patch MLRun Community Edition Deployment images to minikube. Useful if overriding images and running in minikube.
If you have some custom images, before deploying the CE, run:
```
$ python automation/deployment/ce.py patch-minikube-images \
    --mlrun-api-image <MLRUN_API_IMAGE> \
    --mlrun-ui-image <MLRUN_UI_IMAGE> \
    --jupyter-image <JUPYTER_IMAGE>
```

Then you can deploy the CE with:
```
$ python automation/deployment/ce.py deploy \
    --registry-url <REGISTRY_URL> \
    --registry-username <REGISTRY_USERNAME> \
    --registry-password <REGISTRY_PASSWORD> \
    --minikube \
    --override-mlrun-api-image <MLRUN_API_IMAGE> \
    --override-mlrun-ui-image <MLRUN_UI_IMAGE> \
    --override-jupyter-image <JUPYTER_IMAGE>
```
