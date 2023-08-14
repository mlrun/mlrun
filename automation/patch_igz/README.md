# Deploy mlrun-api from your current code on a live system (for debugging)


In order to deploy your current code (for debugging), you need the following:

* Install automation/requirements.txt
* Create a patch_env.yml based on patch_env_template.yml
* Have a docker registry you can push to (e.g. docker.io via account on docker.com) as well as a public mlrun-api repo on it
* Make sure you are logged in into your registry (docker login --username user --password passwd), or optionally add username/password to config
* From mlrun root dir run ./automation/patch_igz/patch_remote.py

WARNING: This may not persist after system restart

Troubleshooting:
* Make sure you created PUBLIC repo named mlrun-api
