# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import expanduser

default_config = {
    "namespace": "",  # default kubernetes namespace
    "dbpath": "",  # db/api url
    # url to nuclio dashboard api (can be with user & token, e.g. https://username:password@dashboard-url.com)
    "nuclio_dashboard_url": "",
    "nuclio_version": "",
    "default_nuclio_runtime": "python:3.7",
    "nest_asyncio_enabled": "",  # enable import of nest_asyncio for corner cases with old jupyter, set "1"
    "ui_url": "",  # remote/external mlrun UI url (for hyperlinks) (This is deprecated in favor of the ui block)
    "remote_host": "",
    "version": "",  # will be set to current version
    "images_tag": "",  # tag to use with mlrun images e.g. mlrun/mlrun (defaults to version)
    "images_registry": "",  # registry to use with mlrun images e.g. quay.io/ (defaults to empty, for dockerhub)
    # comma separated list of images that are in the specified images_registry, and therefore will be enriched with this
    # registry when used. default to mlrun/* which means any image which is of the mlrun repository (mlrun/mlrun,
    # mlrun/ml-base, etc...)
    "images_to_enrich_registry": "^mlrun/*",
    "kfp_ttl": "14400",  # KFP ttl in sec, after that completed PODs will be deleted
    "kfp_image": "",  # image to use for KFP runner (defaults to mlrun/mlrun)
    "dask_kfp_image": "",  # image to use for dask KFP runner (defaults to mlrun/ml-base)
    "igz_version": "",  # the version of the iguazio system the API is running on
    "iguazio_api_url": "",  # the url to iguazio api
    "spark_app_image": "",  # image to use for spark operator app runtime
    "spark_app_image_tag": "",  # image tag to use for spark opeartor app runtime
    "spark_history_server_path": "",  # spark logs directory for spark history server
    "spark_operator_version": "spark-2",  # the version of the spark operator in use
    "builder_alpine_image": "alpine:3.13.1",  # builder alpine image (as kaniko's initContainer)
    "package_path": "mlrun",  # mlrun pip package
    "default_base_image": "mlrun/mlrun",  # default base image when doing .deploy()
    "default_project": "default",  # default project name
    "default_archive": "",  # default remote archive URL (for build tar.gz)
    "mpijob_crd_version": "",  # mpijob crd version (e.g: "v1alpha1". must be in: mlrun.runtime.MPIJobCRDVersions)
    "hub_url": "https://raw.githubusercontent.com/mlrun/functions/{tag}/{name}/function.yaml",
    "ipython_widget": True,
    "log_level": "INFO",
    # log formatter (options: human | json)
    "log_formatter": "human",
    "submit_timeout": "180",  # timeout when submitting a new k8s resource
    # runtimes cleanup interval in seconds
    "runtimes_cleanup_interval": "300",
    # runs monitoring interval in seconds
    "runs_monitoring_interval": "30",
    # the grace period (in seconds) that will be given to runtime resources (after they're in terminal state)
    # before deleting them
    "runtime_resources_deletion_grace_period": "14400",
    "scrape_metrics": True,
    # sets the background color that is used in printed tables in jupyter
    "background_color": "#4EC64B",
    "artifact_path": "",  # default artifacts path/url
    # FIXME: Adding these defaults here so we won't need to patch the "installing component" (provazio-controller) to
    #  configure this values on field systems, for newer system this will be configured correctly
    "v3io_api": "http://v3io-webapi:8081",
    "v3io_framesd": "http://framesd:8080",
    "datastore": {"async_source_mode": "disabled"},
    # default node selector to be applied to all functions - json string base64 encoded format
    "default_function_node_selector": "e30=",
    # default priority class to be applied to functions running on k8s cluster
    "default_function_priority_class_name": "",
    # valid options for priority classes - separated by a comma
    "valid_function_priority_class_names": "",
    "function_defaults": {
        "image_by_kind": {
            "job": "mlrun/mlrun",
            "serving": "mlrun/mlrun",
            "nuclio": "mlrun/mlrun",
            "remote": "mlrun/mlrun",
            "dask": "mlrun/ml-base",
            "mpijob": "mlrun/ml-models",
        }
    },
    "httpdb": {
        "port": 8080,
        "dirpath": expanduser("~/.mlrun/db"),
        "dsn": "sqlite:////mlrun/db/mlrun.db?check_same_thread=false",
        "debug": False,
        "user": "",
        "password": "",
        "token": "",
        "logs_path": "/mlrun/db/logs",
        "data_volume": "",
        "real_path": "",
        "db_type": "sqldb",
        "max_workers": "",
        "db": {"commit_retry_timeout": 30, "commit_retry_interval": 3},
        "jobs": {
            # whether to allow to run local runtimes in the API - configurable to allow the scheduler testing to work
            "allow_local_run": False,
        },
        "authentication": {
            "mode": "none",  # one of none, basic, bearer, iguazio
            "basic": {"username": "", "password": ""},
            "bearer": {"token": ""},
            "iguazio": {
                "session_verification_endpoint": "data_sessions/verifications/app_service",
            },
        },
        "nuclio": {
            # One of ClusterIP | NodePort
            "default_service_type": "NodePort",
            # The following modes apply when user did not configure an ingress
            #
            #   name        |  description
            #  ---------------------------------------------------------------------
            #   never       |  never enrich with an ingress
            #   always      |  always enrich with an ingress, regardless the service type
            #   onClusterIP |  enrich with an ingress only when `mlrun.config.httpdb.nuclio.default_service_type`
            #                  is set to ClusterIP
            #  ---------------------------------------------------------------------
            # Note: adding a mode requires special handling on
            # - mlrun.runtimes.constants.NuclioIngressAddTemplatedIngressModes
            # - mlrun.runtimes.function.enrich_function_with_ingress
            "add_templated_ingress_host_mode": "never",
        },
        "authorization": {
            "mode": "none",  # one of none, opa
            "opa": {
                "address": "",
                "request_timeout": 10,
                "permission_query_path": "",
                "permission_filter_path": "",
                "log_level": 0,
            },
        },
        "scheduling": {
            # the minimum interval that will be allowed between two scheduled jobs - e.g. a job wouldn't be
            # allowed to be scheduled to run more then 2 times in X. Can't be less then 1 minute, "0" to disable
            "min_allowed_interval": "10 minutes",
            "default_concurrency_limit": 1,
            # Firing our jobs include things like creating pods which might not be instant, therefore in the case of
            # multiple schedules scheduled to the same time, there might be delays, the default of the scheduler for
            # misfire_grace_time is 1 second, we do not want jobs not being scheduled because of the delays so setting
            # it to None. the default for coalesce it True just adding it here to be explicit
            "scheduler_config": '{"job_defaults": {"misfire_grace_time": null, "coalesce": true}}',
            # one of enabled, disabled, auto (in which it will be determined by whether the authorization mode is opa)
            "schedule_credentials_secrets_store_mode": "auto",
        },
        "projects": {
            "leader": "mlrun",
            "followers": "",
            # This is used as the interval for the sync loop both when mlrun is leader and follower
            "periodic_sync_interval": "1 minute",
            "counters_cache_ttl": "10 seconds",
            # access key to be used when the leader is iguazio and polling is done from it
            "iguazio_access_key": "",
            # the initial implementation was cache and was working great, now it's not needed because we get (read/list)
            # from leader because of some auth restriction, we will probably go back to it at some point since it's
            # better performance wise, so made it a mode
            # one of: cache, none
            "follower_projects_store_mode": "cache",
            "project_owners_cache_ttl": "30 seconds",
        },
        # The API needs to know what is its k8s svc url so it could enrich it in the jobs it creates
        "api_url": "",
        "builder": {
            # setting the docker registry to be used for built images, can include the repository as well, e.g.
            # index.docker.io/<username>, if not included repository will default to mlrun
            "docker_registry": "",
            "docker_registry_secret": "",
            # the requirement specifier used by the builder when installing mlrun in images when it runs
            # pip install <requirement_specifier>, e.g. mlrun==0.5.4, mlrun~=0.5,
            # git+https://github.com/mlrun/mlrun@development. by default uses the version
            "mlrun_version_specifier": "",
            "kaniko_image": "gcr.io/kaniko-project/executor:v0.24.0",  # kaniko builder image
            "kaniko_init_container_image": "alpine:3.13.1",
            # additional docker build args in json encoded base64 format
            "build_args": "",
        },
        "v3io_api": "",
        "v3io_framesd": "",
    },
    "model_endpoint_monitoring": {
        "serving_stream_args": {"shard_count": 1, "retention_period_hours": 24},
        "drift_thresholds": {"default": {"possible_drift": 0.5, "drift_detected": 0.7}},
        "store_prefixes": {
            "default": "v3io:///users/pipelines/{project}/model-endpoints/{kind}",
            "user_space": "v3io:///projects/{project}/model-endpoints/{kind}",
        },
        "batch_processing_function_branch": "master",
    },
    "secret_stores": {
        "vault": {
            # URLs to access Vault. For example, in a local env (Minikube on Mac) these would be:
            # http://docker.for.mac.localhost:8200
            "url": "",
            "remote_url": "",
            "role": "",
            "token_path": "~/.mlrun/vault",
            "project_service_account_name": "mlrun-vault-{project}",
            "token_ttl": 180000,
            # This config is for debug/testing purposes only!
            "user_token": "",
        },
        "azure_vault": {
            "url": "https://{name}.vault.azure.net",
            "default_secret_name": None,
            "secret_path": "~/.mlrun/azure_vault",
        },
        "kubernetes": {
            # When this is True (the default), all project secrets will be automatically added to each job,
            # unless user asks for a specific list of secrets.
            "auto_add_project_secrets": True,
            "project_secret_name": "mlrun-project-secrets-{project}",
            "env_variable_prefix": "MLRUN_K8S_SECRET__",
        },
    },
    "feature_store": {
        "data_prefixes": {
            "default": "v3io:///projects/{project}/FeatureStore/{name}/{kind}",
            "nosql": "v3io:///projects/{project}/FeatureStore/{name}/{kind}",
        },
        "default_targets": "parquet,nosql",
        "default_job_image": "mlrun/mlrun",
        "flush_interval": None,
    },
    "ui": {
        "projects_prefix": "projects",  # The UI link prefix for projects
        "url": "",  # remote/external mlrun UI url (for hyperlinks)
    },
    "marketplace": {
        "k8s_secrets_project_name": "-marketplace-secrets",
        "catalog_filename": "catalog.json",
        "default_source": {
            # Set to false to avoid creating a global source (for example in a dark site)
            "create": True,
            "name": "mlrun_global_hub",
            "description": "MLRun global function hub",
            "url": "https://raw.githubusercontent.com/mlrun/marketplace",
            "channel": "master",
        },
    },
    "storage": {
        # What type of auto-mount to use for functions. Can be one of: none, auto, v3io_credentials, v3io_fuse, pvc.
        # Default is auto - which is v3io_credentials when running on Iguazio. If not Iguazio: pvc if the
        # MLRUN_PVC_MOUNT env is configured or auto_mount_params contain "pvc_name". Otherwise will do nothing (none).
        "auto_mount_type": "auto",
        # Extra parameters to pass to the mount call (will be passed as kwargs). Parameters can be either:
        # 1. A string of comma-separated parameters, using this format: "param1=value1,param2=value2"
        # 2. A base-64 encoded json dictionary containing the list of parameters
        "auto_mount_params": "",
    },
}
