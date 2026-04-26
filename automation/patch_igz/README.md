# Deploy mlrun-api from your current code on a live system (for debugging)


In order to deploy your current code (for debugging), you need the following:

* Install automation/requirements.txt 
~~~
make install-automation-requirements
~~~
* Create a patch_env.yml based on patch_env_template.yml
* Have a docker registry you can push to (e.g. docker.io via account on docker.com) as well as a public mlrun-api repo on it
* Make sure you are logged in into your registry (docker login --username user --password passwd), or optionally add username/password to config
* From mlrun root dir run ./automation/patch_igz/patch_remote.py
* If requesting to reset DB, DB_USER must be defined in patch_env.yml

## Execution modes

* `ssh` (default): SSH into a data node and run kubectl there.
* `kubectl`: Run kubectl locally against the cluster (no SSH credentials required).

Mode is inferred from `KUBECONFIG` when unset — set `KUBECONFIG` (config field or `--kubeconfig`) and you get kubectl mode; leave it empty and you stay on ssh. To force a mode regardless, set `MODE` in the config or pass `--mode {ssh,kubectl}` (e.g. kubectl mode using the kubectl default `~/.kube/config`).

WARNING: This may not persist after system restart

Troubleshooting:
* Make sure you created PUBLIC repo named mlrun-api
