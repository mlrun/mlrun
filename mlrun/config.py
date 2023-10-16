# Copyright 2023 Iguazio
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
"""
Configuration system.

Configuration can be in either a configuration file specified by
MLRUN_CONFIG_FILE environment variable or by environment variables.

Environment variables are in the format "MLRUN_httpdb__port=8080". This will be
mapped to config.httpdb.port. Values should be in JSON format.
"""

import base64
import binascii
import copy
import json
import os
import typing
from collections.abc import Mapping
from datetime import timedelta
from distutils.util import strtobool
from os.path import expanduser
from threading import Lock

import dotenv
import semver
import yaml

import mlrun.common.schemas
import mlrun.errors

env_prefix = "MLRUN_"
env_file_key = f"{env_prefix}CONFIG_FILE"
_load_lock = Lock()
_none_type = type(None)
default_env_file = os.getenv("MLRUN_DEFAULT_ENV_FILE", "~/.mlrun.env")

default_config = {
    "namespace": "",  # default kubernetes namespace
    "kubernetes": {
        "kubeconfig_path": "",  # local path to kubeconfig file (for development purposes),
        # empty by default as the API already running inside k8s cluster
    },
    "dbpath": "",  # db/api url
    # url to nuclio dashboard api (can be with user & token, e.g. https://username:password@dashboard-url.com)
    "nuclio_dashboard_url": "",
    "nuclio_version": "",
    "default_nuclio_runtime": "python:3.9",
    "nest_asyncio_enabled": "",  # enable import of nest_asyncio for corner cases with old jupyter, set "1"
    "ui_url": "",  # remote/external mlrun UI url (for hyperlinks) (This is deprecated in favor of the ui block)
    "remote_host": "",
    "api_base_version": "v1",
    "version": "",  # will be set to current version
    "images_tag": "",  # tag to use with mlrun images e.g. mlrun/mlrun (defaults to version)
    "images_registry": "",  # registry to use with mlrun images e.g. quay.io/ (defaults to empty, for dockerhub)
    # comma separated list of images that are in the specified images_registry, and therefore will be enriched with this
    # registry when used. default to mlrun/* which means any image which is of the mlrun repository (mlrun/mlrun,
    # mlrun/ml-base, etc...)
    "images_to_enrich_registry": "^mlrun/*",
    "kfp_url": "",
    "kfp_ttl": "14400",  # KFP ttl in sec, after that completed PODs will be deleted
    "kfp_image": "mlrun/mlrun",  # image to use for KFP runner (defaults to mlrun/mlrun)
    "dask_kfp_image": "mlrun/ml-base",  # image to use for dask KFP runner (defaults to mlrun/ml-base)
    "igz_version": "",  # the version of the iguazio system the API is running on
    "iguazio_api_url": "",  # the url to iguazio api
    "spark_app_image": "",  # image to use for spark operator app runtime
    "spark_app_image_tag": "",  # image tag to use for spark operator app runtime
    "spark_history_server_path": "",  # spark logs directory for spark history server
    "spark_operator_version": "spark-3",  # the version of the spark operator in use
    "package_path": "mlrun",  # mlrun pip package
    "default_base_image": "mlrun/mlrun",  # default base image when doing .deploy()
    # template for project default image name. Parameter {name} will be replaced with project name
    "default_project_image_name": ".mlrun-project-image-{name}",
    "default_project": "default",  # default project name
    "default_archive": "",  # default remote archive URL (for build tar.gz)
    "mpijob_crd_version": "",  # mpijob crd version (e.g: "v1alpha1". must be in: mlrun.runtime.MPIJobCRDVersions)
    "ipython_widget": True,
    "log_level": "INFO",
    # log formatter (options: human | json)
    "log_formatter": "human",
    "submit_timeout": "180",  # timeout when submitting a new k8s resource
    # runtimes cleanup interval in seconds
    "runtimes_cleanup_interval": "300",
    # runs monitoring interval in seconds
    "runs_monitoring_interval": "30",
    # runs monitoring debouncing interval in seconds for run with non-terminal state without corresponding k8s resource
    # by default the interval will be - (runs_monitoring_interval * 2 ), if set will override the default
    "runs_monitoring_missing_runtime_resources_debouncing_interval": None,
    # the grace period (in seconds) that will be given to runtime resources (after they're in terminal state)
    # before deleting them (4 hours)
    "runtime_resources_deletion_grace_period": "14400",
    "scrape_metrics": True,
    # sets the background color that is used in printed tables in jupyter
    "background_color": "#4EC64B",
    "artifact_path": "",  # default artifacts path/url
    # Add {{workflow.uid}} to artifact_path unless user specified a path manually
    "enrich_artifact_path_with_workflow_id": True,
    "artifacts": {
        "calculate_hash": True,
        # None is handled as False, reason we set None instead of False is that if the server have set the value to
        # some value while the client didn't change it, the server value will be applied.
        # But if both the server and the client set some value, we want the client to take precedence over the server.
        # By setting the default to None we are able to differentiate between the two cases.
        "generate_target_path_from_artifact_hash": None,
    },
    # FIXME: Adding these defaults here so we won't need to patch the "installing component" (provazio-controller) to
    #  configure this values on field systems, for newer system this will be configured correctly
    "v3io_api": "http://v3io-webapi:8081",
    "redis": {
        "url": "",
        "type": "standalone",  # deprecated.
    },
    "sql": {
        "url": "",
    },
    "v3io_framesd": "http://framesd:8080",
    "datastore": {"async_source_mode": "disabled"},
    # default node selector to be applied to all functions - json string base64 encoded format
    "default_function_node_selector": "e30=",
    # default priority class to be applied to functions running on k8s cluster
    "default_function_priority_class_name": "",
    # valid options for priority classes - separated by a comma
    "valid_function_priority_class_names": "",
    # default path prefix for demo data and models
    "default_samples_path": "https://s3.wasabisys.com/iguazio/",
    # default path for tensorboard logs
    "default_tensorboard_logs_path": "/User/.tensorboard/{{project}}",
    # ";" separated list of notebook cell tag names to ignore e.g. "ignore-this;ignore-that"
    "ignored_notebook_tags": "",
    # when set it will force the local=True in run_function(), set to "auto" will run local if there is no k8s
    "force_run_local": "auto",
    # when set (True or non empty str) it will force the mock=True in deploy_function(),
    # set to "auto" will use mock of Nuclio if not detected (no nuclio_version)
    "mock_nuclio_deployment": "",
    "background_tasks": {
        # enabled / disabled
        "timeout_mode": "enabled",
        # timeout in seconds to wait for background task to be updated / finished by the worker responsible for the task
        "default_timeouts": {
            "operations": {"migrations": "3600", "load_project": "60"},
            "runtimes": {"dask": "600"},
        },
    },
    "function": {
        "spec": {
            "image_pull_secret": {"default": None},
            "security_context": {
                # default security context to be applied to all functions - json string base64 encoded format
                # in camelCase format: {"runAsUser": 1000, "runAsGroup": 3000}
                "default": "e30=",  # encoded empty dict
                # see mlrun.common.schemas.function.SecurityContextEnrichmentModes for available options
                "enrichment_mode": "disabled",
                # default 65534 (nogroup), set to -1 to use the user unix id or
                # function.spec.security_context.pipelines.kfp_pod_user_unix_id for kfp pods
                "enrichment_group_id": 65534,
                "pipelines": {
                    # sets the user id to be used for kfp pods when enrichment mode is not disabled
                    "kfp_pod_user_unix_id": 5,
                },
            },
            "service_account": {"default": None},
        },
    },
    # TODO: function defaults should be moved to the function spec config above
    "function_defaults": {
        "image_by_kind": {
            "job": "mlrun/mlrun",
            "serving": "mlrun/mlrun",
            "nuclio": "mlrun/mlrun",
            "remote": "mlrun/mlrun",
            "dask": "mlrun/ml-base",
            "mpijob": "mlrun/mlrun",
        },
        # see enrich_function_preemption_spec for more info,
        # and mlrun.common.schemas.function.PreemptionModes for available options
        "preemption_mode": "prevent",
    },
    "httpdb": {
        "clusterization": {
            # one of chief/worker
            "role": "chief",
            "chief": {
                # when url is specified, it takes precedence over service and port
                "url": "",
                "service": "mlrun-api-chief",
                "port": 8080,
            },
            "worker": {
                "sync_with_chief": {
                    # enabled / disabled
                    "mode": "enabled",
                    "interval": 15,  # seconds
                },
                "request_timeout": 45,  # seconds
            },
            # see server.api.utils.helpers.ensure_running_on_chief
            "ensure_function_running_on_chief_mode": "enabled",
        },
        "port": 8080,
        "dirpath": expanduser("~/.mlrun/db"),
        "dsn": "sqlite:///db/mlrun.db?check_same_thread=false",
        "old_dsn": "",
        "debug": False,
        "user": "",
        "password": "",
        "token": "",
        "logs_path": "./db/logs",
        # when set, these will replace references to the data_volume with the real_path
        "data_volume": "",
        "real_path": "",
        # comma delimited prefixes of paths allowed through the /files API (v3io & the real_path are always allowed).
        # These paths must be schemas (cannot be used for local files). For example "s3://mybucket,gcs://"
        "allowed_file_paths": "s3://,gcs://,gs://,az://,dbfs://",
        "db_type": "sqldb",
        "max_workers": 64,
        # See mlrun.common.schemas.APIStates for options
        "state": "online",
        "retry_api_call_on_exception": "enabled",
        "http_connection_timeout_keep_alive": 11,
        "db": {
            "commit_retry_timeout": 30,
            "commit_retry_interval": 3,
            "conflict_retry_timeout": 15,
            "conflict_retry_interval": None,
            # Whether to perform data migrations on initialization. enabled or disabled
            "data_migrations_mode": "enabled",
            # Whether to perform database migration from sqlite to mysql on initialization
            "database_migration_mode": "enabled",
            "backup": {
                # Whether to use db backups on initialization
                "mode": "enabled",
                "file_format": "db_backup_%Y%m%d%H%M.db",
                "use_rotation": True,
                "rotation_limit": 3,
                # default is 16MB, max 1G, for more info https://dev.mysql.com/doc/refman/8.0/en/packet-too-large.html
                "max_allowed_packet": 64000000,  # 64MB
            },
            # None will set this to be equal to the httpdb.max_workers
            "connections_pool_size": None,
            "connections_pool_max_overflow": None,
            # below is a db-specific configuration
            "mysql": {
                # comma separated mysql modes (globally) to set on runtime
                # optional values (as per https://dev.mysql.com/doc/refman/8.0/en/sql-mode.html#sql-mode-full):
                #
                # if set to "nil" or "none", nothing would be set
                "modes": "STRICT_TRANS_TABLES",
            },
        },
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
            "explicit_ack": "enabled",
        },
        "logs": {
            "decode": {
                # Replace with a replacement marker. Uses � (U+FFFD, the official REPLACEMENT CHARACTER).
                # see https://docs.python.org/3/library/codecs.html#error-handlers for more info and options
                "errors": "replace",
            },
            "pipelines": {
                # pull state mode was introduced to have a way to pull the state of a run which was spawned by a
                # pipeline step instead of pulling the state by getting the run logs
                "pull_state": {
                    # enabled - pull state of a run every "pull_state_interval" seconds and pull logs every
                    # "pull_logs_interval" seconds
                    # disabled - pull logs every "pull_logs_default_interval" seconds
                    "mode": "disabled",
                    # those params are used when mode is enabled
                    "pull_logs_interval": 30,  # seconds
                    "pull_state_interval": 5,  # seconds
                },
            },
            "nuclio": {
                # setting interval to a higher interval than regular jobs / build, because pulling the retrieved logs
                # from nuclio for the deploy status doesn't include the actual live "builder" container logs, but
                # rather a high level status
                "pull_deploy_status_default_interval": 10  # seconds
            },
            # this is the default interval period for pulling logs, if not specified different timeout interval
            "pull_logs_default_interval": 3,  # seconds
            "pull_logs_backoff_no_logs_default_interval": 10,  # seconds
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
            # allowed to be scheduled to run more than 2 times in X. Can't be less than 1 minute, "0" to disable
            "min_allowed_interval": "10 minutes",
            "default_concurrency_limit": 1,
            # Firing our jobs include things like creating pods which might not be instant, therefore in the case of
            # multiple schedules scheduled to the same time, there might be delays, the default of the scheduler for
            # misfire_grace_time is 1 second, we do not want jobs not being scheduled because of the delays so setting
            # it to None. the default for coalesce it True just adding it here to be explicit
            "scheduler_config": '{"job_defaults": {"misfire_grace_time": null, "coalesce": true}}',
        },
        "projects": {
            "leader": "mlrun",
            "retry_leader_request_on_exception": "enabled",
            "followers": "",
            # This is used as the interval for the sync loop both when mlrun is leader and follower
            "periodic_sync_interval": "1 minute",
            "counters_cache_ttl": "2 minutes",
            # access key to be used when the leader is iguazio and polling is done from it
            "iguazio_access_key": "",
            "iguazio_list_projects_default_page_size": 200,
            "project_owners_cache_ttl": "30 seconds",
        },
        # The API needs to know what is its k8s svc url so it could enrich it in the jobs it creates
        "api_url": "",
        "builder": {
            # setting the docker registry to be used for built images, can include the repository as well, e.g.
            # index.docker.io/<username>, if not included repository will default to mlrun
            "docker_registry": "",
            # dockerconfigjson type secret to attach to kaniko pod.
            # For amazon ECR, the secret is expected to provide AWS credentials. Leave empty to use EC2 IAM policy.
            # https://github.com/GoogleContainerTools/kaniko#pushing-to-amazon-ecr
            "docker_registry_secret": "",
            # whether to allow the docker registry we're pulling from to be insecure. "enabled", "disabled" or "auto"
            # which will resolve by the existence of secret
            "insecure_pull_registry_mode": "auto",
            # whether to allow the docker registry we're pushing to, to be insecure. "enabled", "disabled" or "auto"
            # which will resolve by the existence of secret
            "insecure_push_registry_mode": "auto",
            # the requirement specifier used by the builder when installing mlrun in images when it runs
            # pip install <requirement_specifier>, e.g. mlrun==0.5.4, mlrun~=0.5,
            # git+https://github.com/mlrun/mlrun@development. by default uses the version
            "mlrun_version_specifier": "",
            "kaniko_image": "gcr.io/kaniko-project/executor:v1.8.0",  # kaniko builder image
            "kaniko_init_container_image": "alpine:3.18",
            # image for kaniko init container when docker registry is ECR
            "kaniko_aws_cli_image": "amazon/aws-cli:2.7.10",
            # kaniko sometimes fails to get filesystem from image, this is a workaround to retry the process
            # a known issue in Kaniko - https://github.com/GoogleContainerTools/kaniko/issues/1717
            "kaniko_image_fs_extraction_retries": "3",
            # kaniko sometimes fails to push image to registry due to network issues
            "kaniko_image_push_retry": "3",
            # additional docker build args in json encoded base64 format
            "build_args": "",
            "pip_ca_secret_name": "",
            "pip_ca_secret_key": "",
            "pip_ca_path": "/etc/ssl/certs/mlrun/pip-ca-certificates.crt",
            # template for the prefix that the function target image will be enforced to have (as long as it's targeted
            # to be in the configured registry). Supported template values are: {project} {name}
            "function_target_image_name_prefix_template": "func-{project}-{name}",
            "pip_version": "~=23.0",
        },
        "v3io_api": "",
        "v3io_framesd": "",
        # If running from sdk and MLRUN_DBPATH is not set, the db will fallback to a nop db which will not preform any
        # run db operations.
        "nop_db": {
            # if set to true, will raise an error for trying to use run db functionality
            # if set to false, will use a nop db which will not preform any run db operations
            "raise_error": False,
            # if set to true, will log a warning for trying to use run db functionality while in nop db mode
            "verbose": True,
        },
    },
    "model_endpoint_monitoring": {
        "serving_stream_args": {"shard_count": 1, "retention_period_hours": 24},
        "application_stream_args": {"shard_count": 3, "retention_period_hours": 24},
        "drift_thresholds": {"default": {"possible_drift": 0.5, "drift_detected": 0.7}},
        # Store prefixes are used to handle model monitoring storing policies based on project and kind, such as events,
        # stream, and endpoints.
        "store_prefixes": {
            "default": "v3io:///users/pipelines/{project}/model-endpoints/{kind}",
            "user_space": "v3io:///projects/{project}/model-endpoints/{kind}",
            "stream": "",
        },
        # Offline storage path can be either relative or a full path. This path is used for general offline data
        # storage such as the parquet file which is generated from the monitoring stream function for the drift analysis
        "offline_storage_path": "model-endpoints/{kind}",
        # Default http path that points to the monitoring stream nuclio function. Will be used as a stream path
        # when the user is working in CE environment and has not provided any stream path.
        "default_http_sink": "http://nuclio-{project}-model-monitoring-stream.mlrun.svc.cluster.local:8080",
        "default_http_sink_app": "http://nuclio-{project}-{application_name}.mlrun.svc.cluster.local:8080",
        "batch_processing_function_branch": "master",
        "parquet_batching_max_events": 10000,
        "parquet_batching_timeout_secs": timedelta(minutes=30).total_seconds(),
        # See mlrun.model_monitoring.stores.ModelEndpointStoreType for available options
        "store_type": "v3io-nosql",
        "endpoint_store_connection": "",
    },
    "secret_stores": {
        # Use only in testing scenarios (such as integration tests) to avoid using k8s for secrets (will use in-memory
        # "secrets")
        "test_mode_mock_secrets": False,
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
            "auth_secret_name": "mlrun-auth-secrets.{hashed_access_key}",
            "env_variable_prefix": "MLRUN_K8S_SECRET__",
            "global_function_env_secret_name": None,
        },
    },
    "feature_store": {
        "data_prefixes": {
            "default": "v3io:///projects/{project}/FeatureStore/{name}/{kind}",
            "nosql": "v3io:///projects/{project}/FeatureStore/{name}/{kind}",
            # "authority" is optional and generalizes [userinfo "@"] host [":" port]
            "redisnosql": "redis://{authority}/projects/{project}/FeatureStore/{name}/{kind}",
        },
        "default_targets": "parquet,nosql",
        "default_job_image": "mlrun/mlrun",
        "flush_interval": None,
    },
    "ui": {
        "projects_prefix": "projects",  # The UI link prefix for projects
        "url": "",  # remote/external mlrun UI url (for hyperlinks)
    },
    "hub": {
        "k8s_secrets_project_name": "-hub-secrets",
        "catalog_filename": "catalog.json",
        "default_source": {
            # Set false to avoid creating a global source (for example in a dark site)
            "create": True,
            "name": "default",
            "description": "MLRun global function hub",
            "url": "https://mlrun.github.io/marketplace",
            "object_type": "functions",
            "channel": "master",
        },
    },
    "storage": {
        # What type of auto-mount to use for functions. One of: none, auto, v3io_credentials, v3io_fuse, pvc, s3, env.
        # Default is auto - which is v3io_credentials when running on Iguazio. If not Iguazio: pvc if the
        # MLRUN_PVC_MOUNT env is configured or auto_mount_params contain "pvc_name". Otherwise will do nothing (none).
        "auto_mount_type": "auto",
        # Extra parameters to pass to the mount call (will be passed as kwargs). Parameters can be either:
        # 1. A string of comma-separated parameters, using this format: "param1=value1,param2=value2"
        # 2. A base-64 encoded json dictionary containing the list of parameters
        "auto_mount_params": "",
        # map file data items starting with virtual path to the real path, used when consumers have different mounts
        # e.g. Windows client (on host) and Linux container (Jupyter, Nuclio..) need to access the same files/artifacts
        # need to map container path to host windows paths, e.g. "\data::c:\\mlrun_data" ("::" used as splitter)
        "item_to_real_path": "",
    },
    "default_function_pod_resources": {
        "requests": {"cpu": None, "memory": None, "gpu": None},
        "limits": {"cpu": None, "memory": None, "gpu": None},
    },
    "default_spark_resources": {
        "driver": {
            "requests": {"cpu": "1", "memory": "2g"},
            "limits": {"cpu": "2", "memory": "2g"},
        },
        "executor": {
            "requests": {"cpu": "1", "memory": "5g"},
            "limits": {"cpu": "2", "memory": "5g"},
        },
    },
    # preemptible node selector and tolerations to be added when running on spot nodes
    "preemptible_nodes": {
        # encoded empty dict
        "node_selector": "e30=",
        # encoded empty list
        "tolerations": "W10=",
    },
    "http_retry_defaults": {
        "max_retries": 3,
        "backoff_factor": 1,
        "status_codes": [500, 502, 503, 504],
    },
    "ce": {
        # ce mode can be one of: "", lite, full
        "mode": "",
        # not possible to call this "version" because the Config class has a "version" property
        # which returns the version from the version.json file
        "release": "",
    },
    "debug": {
        "expose_internal_api_endpoints": False,
    },
    "workflows": {
        "default_workflow_runner_name": "workflow-runner-{}",
        # Default timeout seconds for retrieving workflow id after execution:
        "timeouts": {"local": 120, "kfp": 30, "remote": 30},
    },
    "log_collector": {
        "address": "localhost:8282",
        # log collection mode can be one of: "sidecar", "legacy", "best-effort"
        # "sidecar" - use the sidecar to collect logs
        # "legacy" - use the legacy log collection method (logs are collected straight from the pod)
        # "best-effort" - use the sidecar, but if for some reason it's not available use the legacy method
        # note that this mode also effects the log querying method as well, meaning if the mode is "best-effort"
        # the log query will try to use the sidecar first and if it's not available it will use the legacy method
        # TODO: once this is changed to "sidecar" by default, also change in common_fixtures.py
        "mode": "legacy",
        # interval for collecting and sending runs which require their logs to be collected
        "periodic_start_log_interval": 10,
        "failed_runs_grace_period": 3600,
        "verbose": True,
        # the number of workers which will be used to trigger the start log collection
        "concurrent_start_logs_workers": 15,
        # the time in hours in which to start log collection from.
        # after upgrade, we might have runs which completed in the mean time or still in non-terminal state and
        # we want to collect their logs in the new log collection method (sidecar)
        # default is 4 hours = 4*60*60 = 14400 seconds
        "api_downtime_grace_period": 14400,
        "get_logs": {
            # the number of retries to get logs from the log collector
            "max_retries": 3,
        },
        # interval for stopping log collection for runs which are in a terminal state
        "stop_logs_interval": 3600,
    },
    # Configurations for the `mlrun.package` sub-package involving packagers - logging returned outputs and parsing
    # inputs data items:
    "packagers": {
        # Whether to enable packagers. True will wrap each run in the `mlrun.package.handler` decorator to log and parse
        # using packagers.
        "enabled": True,
        # Whether to treat returned tuples from functions as a tuple and not as multiple returned items. If True, all
        # returned values will be packaged together as the tuple they are returned in. Default is False to enable
        # logging multiple returned items.
        "pack_tuples": False,
        # In multi-workers run, only the logging worker will pack the outputs and log the results and artifacts.
        # Otherwise, the workers will log the results and artifacts using the same keys, overriding them. It is common
        # that only the main worker (usualy rank 0) will log, so this is the default value.
        "logging_worker": 0,
        # TODO: Consider adding support for logging from all workers (ignoring the `logging_worker`) and add the worker
        #       number to the artifact / result key (like "<key>-rank<#>". Results can have reduce operation in the
        #       log hint to average / min / max them across all the workers (default operation should be average).
    },
    # Events are currently (and only) used to audit changes and record access to MLRun entities (such as secrets)
    "events": {
        # supported modes "enabled", "disabled".
        # "enabled" - events are emitted.
        # "disabled" - a nop client is used (aka doing nothing).
        "mode": "disabled",
        "verbose": False,
        # used for igz client when emitting events
        "access_key": "",
    },
}

_is_running_as_api = None


def is_running_as_api():
    # MLRUN_IS_API_SERVER is set when running the api server which is being done through the CLI command mlrun db
    global _is_running_as_api

    if _is_running_as_api is None:
        _is_running_as_api = os.getenv("MLRUN_IS_API_SERVER", "false").lower() == "true"

    return _is_running_as_api


class Config:
    _missing = object()

    def __init__(self, cfg=None):
        cfg = {} if cfg is None else cfg

        # Can't use self._cfg = cfg → infinite recursion
        object.__setattr__(self, "_cfg", cfg)

    def __getattr__(self, attr):
        val = self._cfg.get(attr, self._missing)
        if val is self._missing:
            raise AttributeError(attr)

        if isinstance(val, Mapping):
            return self.__class__(val)
        return val

    def __setattr__(self, attr, value):
        # in order for the dbpath setter to work
        if attr == "dbpath":
            super().__setattr__(attr, value)
        else:
            self._cfg[attr] = value

    def __dir__(self):
        return list(self._cfg) + dir(self.__class__)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({self._cfg!r})"

    def update(self, cfg, skip_errors=False):
        for key, value in cfg.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    try:
                        setattr(self, key, value)
                    except mlrun.errors.MLRunRuntimeError as exc:
                        if not skip_errors:
                            raise exc
                        print(
                            f"Warning, failed to set config key {key}={value}, {mlrun.errors.err_to_str(exc)}"
                        )

    def dump_yaml(self, stream=None):
        return yaml.dump(self._cfg, stream, default_flow_style=False)

    @classmethod
    def from_dict(cls, dict_):
        return cls(copy.deepcopy(dict_))

    @staticmethod
    def get_build_args():
        build_args = {}
        if config.httpdb.builder.build_args:
            build_args_json = base64.b64decode(
                config.httpdb.builder.build_args
            ).decode()
            build_args = json.loads(build_args_json)

        return build_args

    @staticmethod
    def is_pip_ca_configured():
        return (
            config.httpdb.builder.pip_ca_secret_name
            and config.httpdb.builder.pip_ca_secret_key
            and config.httpdb.builder.pip_ca_path
        )

    @staticmethod
    def get_default_hub_source() -> str:
        default_source = config.hub.default_source
        return f"{default_source.url}/{default_source.object_type}/{default_source.channel}/"

    @staticmethod
    def decode_base64_config_and_load_to_object(
        attribute_path: str, expected_type=dict
    ):
        """
        decodes and loads the config attribute to expected type
        :param attribute_path: the path in the default_config e.g. preemptible_nodes.node_selector
        :param expected_type: the object type valid values are : `dict`, `list` etc...
        :return: the expected type instance
        """
        attributes = attribute_path.split(".")
        raw_attribute_value = config
        for part in attributes:
            try:
                raw_attribute_value = raw_attribute_value.__getattr__(part)
            except AttributeError:
                raise mlrun.errors.MLRunNotFoundError(
                    "Attribute does not exist in config"
                )
        # There is a bug in the installer component in iguazio system that causes the configured value to be base64 of
        # null (without conditioning it we will end up returning None instead of empty dict)
        if raw_attribute_value and raw_attribute_value != "bnVsbA==":
            try:
                decoded_attribute_value = base64.b64decode(raw_attribute_value).decode()
            except Exception:
                raise mlrun.errors.MLRunInvalidArgumentTypeError(
                    f"Unable to decode {attribute_path}"
                )
            parsed_attribute_value = json.loads(decoded_attribute_value)
            if type(parsed_attribute_value) != expected_type:
                raise mlrun.errors.MLRunInvalidArgumentTypeError(
                    f"Expected type {expected_type}, got {type(parsed_attribute_value)}"
                )
            return parsed_attribute_value
        return expected_type()

    def get_default_function_node_selector(self) -> dict:
        return self.decode_base64_config_and_load_to_object(
            "default_function_node_selector", dict
        )

    def get_preemptible_node_selector(self) -> dict:
        return self.decode_base64_config_and_load_to_object(
            "preemptible_nodes.node_selector", dict
        )

    def get_preemptible_tolerations(self) -> list:
        return self.decode_base64_config_and_load_to_object(
            "preemptible_nodes.tolerations", list
        )

    def get_default_function_security_context(self) -> dict:
        return self.decode_base64_config_and_load_to_object(
            "function.spec.security_context.default", dict
        )

    def is_preemption_nodes_configured(self):
        if (
            not self.get_preemptible_tolerations()
            and not self.get_preemptible_node_selector()
        ):
            return False
        return True

    @staticmethod
    def get_valid_function_priority_class_names():
        valid_function_priority_class_names = []
        if not config.valid_function_priority_class_names:
            return valid_function_priority_class_names

        # Manually ensure we have only unique values because we want to keep the order and using a set would lose it
        for priority_class_name in config.valid_function_priority_class_names.split(
            ","
        ):
            if priority_class_name not in valid_function_priority_class_names:
                valid_function_priority_class_names.append(priority_class_name)
        return valid_function_priority_class_names

    @staticmethod
    def is_running_on_iguazio() -> bool:
        return config.igz_version is not None and config.igz_version != ""

    @staticmethod
    def get_security_context_enrichment_group_id(user_unix_id: int) -> int:
        enrichment_group_id = int(
            config.function.spec.security_context.enrichment_group_id
        )

        # if enrichment group id is -1 we set group id to user unix id
        if enrichment_group_id == -1:
            if user_unix_id is None:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "User unix id is required to populate group id when enrichment group id is -1."
                    "See mlrun.config.function.spec.security_context.enrichment_group_id for more details."
                )
            return user_unix_id

        return enrichment_group_id

    @staticmethod
    def get_parsed_igz_version() -> typing.Optional[semver.VersionInfo]:
        if not config.igz_version:
            return None
        try:
            parsed_version = semver.VersionInfo.parse(config.igz_version)
            return parsed_version
        except ValueError:
            # iguazio version is semver compatible only from 3.2, before that it will be something
            # like 3.0_b177_20210806003728
            semver_compatible_igz_version = config.igz_version.split("_")[0]
            return semver.VersionInfo.parse(f"{semver_compatible_igz_version}.0")

    def verify_security_context_enrichment_mode_is_allowed(self):
        if (
            config.function.spec.security_context.enrichment_mode
            == mlrun.common.schemas.function.SecurityContextEnrichmentModes.disabled
        ):
            return

        igz_version = self.get_parsed_igz_version()
        if not igz_version:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Unable to determine if security context enrichment mode is allowed. Missing iguazio version"
            )

        if igz_version < semver.VersionInfo.parse("3.5.1-b1"):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Security context enrichment mode enabled (override/retain) "
                f"is not allowed for iguazio version: {igz_version} < 3.5.1"
            )

    def resolve_kfp_url(self, namespace=None):
        if config.kfp_url:
            return config.kfp_url
        igz_version = self.get_parsed_igz_version()
        # TODO: When Iguazio 3.4 will deprecate we can remove this line
        if igz_version and igz_version <= semver.VersionInfo.parse("3.6.0-b1"):
            if namespace is None:
                if not config.namespace:
                    raise mlrun.errors.MLRunNotFoundError(
                        "For KubeFlow Pipelines to function, a namespace must be configured"
                    )
                namespace = config.namespace
            # When instead of host we provided namespace we tackled this issue
            # https://github.com/canonical/bundle-kubeflow/issues/412
            # TODO: When we'll move to kfp 1.4.0 (server side) it should be resolved
            return f"http://ml-pipeline.{namespace}.svc.cluster.local:8888"
        return None

    def resolve_chief_api_url(self) -> str:
        if self.httpdb.clusterization.chief.url:
            return self.httpdb.clusterization.chief.url
        if not self.httpdb.clusterization.chief.service:
            raise mlrun.errors.MLRunNotFoundError(
                "For resolving chief url, chief service name must be provided"
            )
        if self.namespace is None:
            raise mlrun.errors.MLRunNotFoundError(
                "For resolving chief url, namespace must be provided"
            )

        chief_api_url = f"http://{self.httpdb.clusterization.chief.service}.{self.namespace}.svc.cluster.local"
        if config.httpdb.clusterization.chief.port:
            chief_api_url = f"{chief_api_url}:{self.httpdb.clusterization.chief.port}"

        self.httpdb.clusterization.chief.url = chief_api_url
        return self.httpdb.clusterization.chief.url

    @staticmethod
    def get_storage_auto_mount_params():
        auto_mount_params = {}
        if config.storage.auto_mount_params:
            try:
                auto_mount_params = base64.b64decode(
                    config.storage.auto_mount_params, validate=True
                ).decode()
                auto_mount_params = json.loads(auto_mount_params)
            except binascii.Error:
                # Importing here to avoid circular dependencies
                from .utils import list2dict

                # String wasn't base64 encoded. Parse it using a 'p1=v1,p2=v2' format.
                mount_params = config.storage.auto_mount_params.split(",")
                auto_mount_params = list2dict(mount_params)
        if not isinstance(auto_mount_params, dict):
            raise TypeError(
                f"data in storage.auto_mount_params does not resolve to a dictionary: {auto_mount_params}"
            )

        return auto_mount_params

    def get_default_function_pod_resources(
        self, with_gpu_requests=False, with_gpu_limits=False
    ):
        resources = {}
        resource_requirements = ["requests", "limits"]
        for requirement in resource_requirements:
            with_gpu = (
                with_gpu_requests if requirement == "requests" else with_gpu_limits
            )
            resources[
                requirement
            ] = self.get_default_function_pod_requirement_resources(
                requirement, with_gpu
            )
        return resources

    def resolve_runs_monitoring_missing_runtime_resources_debouncing_interval(self):
        return (
            float(self.runs_monitoring_missing_runtime_resources_debouncing_interval)
            if self.runs_monitoring_missing_runtime_resources_debouncing_interval
            else float(config.runs_monitoring_interval) * 2.0
        )

    @staticmethod
    def get_default_function_pod_requirement_resources(
        requirement: str, with_gpu: bool = True
    ):
        """
        :param requirement: kubernetes requirement resource one of the following : requests, limits
        :param with_gpu: whether to return requirement resources with nvidia.com/gpu field (e.g. you cannot specify
         GPU requests without specifying GPU limits) https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
        :return: a dict containing the defaults resources (cpu, memory, nvidia.com/gpu)
        """
        resources: dict = copy.deepcopy(config.default_function_pod_resources.to_dict())
        gpu_type = "nvidia.com/gpu"
        gpu = "gpu"
        resource_requirement = resources.get(requirement, {})
        resource_requirement.setdefault(gpu)
        if with_gpu:
            resource_requirement[gpu_type] = resource_requirement.pop(gpu)
        else:
            resource_requirement.pop(gpu)
        return resource_requirement

    def to_dict(self):
        return copy.deepcopy(self._cfg)

    @staticmethod
    def reload():
        _populate()

    @property
    def version(self):
        # importing here to avoid circular dependency
        from mlrun.utils.version import Version

        return Version().get()["version"]

    @staticmethod
    def resolve_ui_url():
        # ui_url is deprecated in favor of the ui.url (we created the ui block)
        # since the config class is used in a "recursive" way, we can't use property like we used in other places
        # since the property will need to be url, which exists in other structs as well
        return config.ui.url or config.ui_url

    @property
    def dbpath(self):
        return self._dbpath

    @dbpath.setter
    def dbpath(self, value):
        self._dbpath = value
        if value:
            # importing here to avoid circular dependency
            import mlrun.db

            # when dbpath is set we want to connect to it which will sync configuration from it to the client
            mlrun.db.get_run_db(value, force_reconnect=True)

    def is_api_running_on_k8s(self):
        # determine if the API service is attached to K8s cluster
        # when there is a cluster the .namespace is set
        return bool(mlrun.mlconf.namespace)

    def is_nuclio_detected(self):
        # determine is Nuclio service is detected, when the nuclio_version is not set
        return bool(mlrun.mlconf.nuclio_version)

    def use_nuclio_mock(self, force_mock=None):
        # determine if to use Nuclio mock service
        mock_nuclio = mlrun.mlconf.mock_nuclio_deployment
        if mock_nuclio and mock_nuclio == "auto":
            mock_nuclio = not mlrun.mlconf.is_nuclio_detected()
        return True if mock_nuclio and force_mock is None else force_mock

    def get_v3io_access_key(self):
        # Get v3io access key from the environment
        return os.environ.get("V3IO_ACCESS_KEY")

    def get_model_monitoring_file_target_path(
        self,
        project: str = "",
        kind: str = "",
        target: str = "online",
        artifact_path: str = None,
        application_name: str = None,
    ) -> str:
        """Get the full path from the configuration based on the provided project and kind.

        :param project:         Project name.
        :param kind:            Kind of target path (e.g. events, log_stream, endpoints, etc.)
        :param target:          Can be either online or offline. If the target is online, then we try to get a specific
                                path for the provided kind. If it doesn't exist, use the default path.
                                If the target path is offline and the offline path is already a full path in the
                                configuration, then the result will be that path as-is. If the offline path is a
                                relative path, then the result will be based on the project artifact path and the
                                offline relative path. If project artifact path wasn't provided, then we use MLRun
                                artifact path instead.
        :param artifact_path:   Optional artifact path that will be used as a relative path. If not provided, the
                                relative artifact path will be taken from the global MLRun artifact path.
        :param application_name:Application name, None for model_monitoring_stream.

        :return: Full configured path for the provided kind.
        """

        if target != "offline":
            store_prefix_dict = (
                mlrun.mlconf.model_endpoint_monitoring.store_prefixes.to_dict()
            )
            if store_prefix_dict.get(kind):
                # Target exist in store prefix and has a valid string value
                return store_prefix_dict[kind].format(project=project)

            if (
                application_name
                != mlrun.common.schemas.model_monitoring.constants.MonitoringFunctionNames.STREAM
            ):
                return mlrun.mlconf.model_endpoint_monitoring.store_prefixes.user_space.format(
                    project=project,
                    kind=kind
                    if application_name is None
                    else f"{kind}-{application_name.lower()}",
                )
            return mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
                project=project,
                kind=kind
                if application_name is None
                else f"{kind}-{application_name.lower()}",
            )

        # Get the current offline path from the configuration
        file_path = mlrun.mlconf.model_endpoint_monitoring.offline_storage_path.format(
            project=project, kind=kind
        )

        # Absolute path
        if any(value in file_path for value in ["://", ":///"]) or os.path.isabs(
            file_path
        ):
            return file_path

        # Relative path
        else:
            artifact_path = artifact_path or config.artifact_path
            if artifact_path[-1] != "/":
                artifact_path += "/"

            return mlrun.utils.helpers.fill_project_path_template(
                artifact_path=artifact_path + file_path, project=project
            )

    def is_ce_mode(self) -> bool:
        # True if the setup is in CE environment
        return isinstance(mlrun.mlconf.ce, mlrun.config.Config) and any(
            ver in mlrun.mlconf.ce.mode for ver in ["lite", "full"]
        )

    def get_s3_storage_options(self) -> typing.Dict[str, typing.Any]:
        """
        Generate storage options dictionary as required for handling S3 path in fsspec. The model monitoring stream
        graph uses this method for generating the storage options for S3 parquet target path.
        :return: A storage options dictionary in which each key-value pair  represents a particular configuration,
        such as endpoint_url or aws access key.
        """
        key = mlrun.get_secret_or_env("AWS_ACCESS_KEY_ID")
        secret = mlrun.get_secret_or_env("AWS_SECRET_ACCESS_KEY")

        force_non_anonymous = mlrun.get_secret_or_env("S3_NON_ANONYMOUS")
        profile = mlrun.get_secret_or_env("AWS_PROFILE")

        storage_options = dict(
            anon=not (force_non_anonymous or (key and secret)),
            key=key,
            secret=secret,
        )

        endpoint_url = mlrun.get_secret_or_env("S3_ENDPOINT_URL")
        if endpoint_url:
            client_kwargs = {"endpoint_url": endpoint_url}
            storage_options["client_kwargs"] = client_kwargs

        if profile:
            storage_options["profile"] = profile

        return storage_options

    def is_explicit_ack(self) -> bool:
        return self.httpdb.nuclio.explicit_ack == "enabled" and (
            not self.nuclio_version or self.nuclio_version >= "1.11.20"
        )


# Global configuration
config = Config.from_dict(default_config)


def _populate(skip_errors=False):
    """Populate configuration from config file (if exists in environment) and
    from environment variables.

    populate will run only once, after first call it does nothing.
    """
    global _loaded

    with _load_lock:
        _do_populate(skip_errors=skip_errors)


def _do_populate(env=None, skip_errors=False):
    global config

    if not os.environ.get("MLRUN_IGNORE_ENV_FILE"):
        if "MLRUN_ENV_FILE" in os.environ:
            env_file = os.path.expanduser(os.environ["MLRUN_ENV_FILE"])
            dotenv.load_dotenv(env_file, override=True)
        else:
            env_file = os.path.expanduser(default_env_file)
            if os.path.isfile(env_file):
                dotenv.load_dotenv(env_file, override=True)

    if not config:
        config = Config.from_dict(default_config)
    else:
        config.update(default_config)
    config_path = os.environ.get(env_file_key)
    if config_path:
        with open(config_path) as fp:
            data = yaml.safe_load(fp)

        if not isinstance(data, dict):
            raise TypeError(f"configuration in {config_path} not a dict")

        config.update(data, skip_errors=skip_errors)

    data = read_env(env)
    if data:
        config.update(data, skip_errors=skip_errors)

    _validate_config(config)


def _validate_config(config):
    try:
        limits_gpu = config.default_function_pod_resources.limits.gpu
        requests_gpu = config.default_function_pod_resources.requests.gpu
        _verify_gpu_requests_and_limits(
            requests_gpu=requests_gpu,
            limits_gpu=limits_gpu,
        )
    except AttributeError:
        pass

    config.verify_security_context_enrichment_mode_is_allowed()


def _verify_gpu_requests_and_limits(requests_gpu: str = None, limits_gpu: str = None):
    # https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
    if requests_gpu and not limits_gpu:
        raise mlrun.errors.MLRunConflictError(
            "You cannot specify GPU requests without specifying limits"
        )
    if requests_gpu and limits_gpu and requests_gpu != limits_gpu:
        raise mlrun.errors.MLRunConflictError(
            f"When specifying both GPU requests and limits these two values must be equal, "
            f"requests_gpu={requests_gpu}, limits_gpu={limits_gpu}"
        )


def _convert_resources_to_str(config: dict = None):
    resources_types = ["cpu", "memory", "gpu"]
    resource_requirements = ["requests", "limits"]
    if not config.get("default_function_pod_resources"):
        return
    for requirement in resource_requirements:
        resource_requirement = config.get("default_function_pod_resources").get(
            requirement
        )
        if not resource_requirement:
            continue
        for resource_type in resources_types:
            value = resource_requirement.setdefault(resource_type, None)
            if value is None:
                continue
            resource_requirement[resource_type] = str(value)


def _convert_str(value, typ):
    if typ in (str, _none_type):
        return value

    if typ is bool:
        return strtobool(value)

    # e.g. int('8080') → 8080
    return typ(value)


def read_env(env=None, prefix=env_prefix):
    """Read configuration from environment"""
    env = os.environ if env is None else env

    config = {}
    for key, value in env.items():
        if not key.startswith(env_prefix) or key == env_file_key:
            continue
        try:
            value = json.loads(value)  # values can be JSON encoded
        except ValueError:
            pass  # Leave as string
        key = key[len(env_prefix) :]  # Trim MLRUN_
        path = key.lower().split("__")  # 'A__B' → ['a', 'b']
        cfg = config
        while len(path) > 1:
            name, *path = path
            cfg = cfg.setdefault(name, {})
        cfg[path[0]] = value

    env_dbpath = env.get("MLRUN_DBPATH", "")
    # expected format: https://mlrun-api.tenant.default-tenant.app.some-system.some-namespace.com
    is_remote_mlrun = (
        env_dbpath.startswith("https://mlrun-api.") and "tenant." in env_dbpath
    )

    # It's already a standard to set this env var to configure the v3io api, so we're supporting it (instead
    # of MLRUN_V3IO_API), in remote usage this can be auto detected from the DBPATH
    v3io_api = env.get("V3IO_API")
    if v3io_api:
        config["v3io_api"] = v3io_api
    elif is_remote_mlrun:
        # in remote mlrun we can't use http, so we'll use https
        config["v3io_api"] = env_dbpath.replace("https://mlrun-api.", "https://webapi.")

    # It's already a standard to set this env var to configure the v3io framesd, so we're supporting it (instead
    # of MLRUN_V3IO_FRAMESD), in remote usage this can be auto detected from the DBPATH
    v3io_framesd = env.get("V3IO_FRAMESD")
    if v3io_framesd:
        config["v3io_framesd"] = v3io_framesd
    elif is_remote_mlrun:
        config["v3io_framesd"] = env_dbpath.replace(
            "https://mlrun-api.", "https://framesd."
        )

    uisvc = env.get("MLRUN_UI_SERVICE_HOST")
    igz_domain = env.get("IGZ_NAMESPACE_DOMAIN")

    # workaround to try and detect IGZ domain
    if not igz_domain and "MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY" in env:
        registry = env["MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY"]
        if registry.startswith("docker-registry.default-tenant"):
            igz_domain = registry[len("docker-registry.") :]
            if ":" in igz_domain:
                igz_domain = igz_domain[: igz_domain.rfind(":")]
            env["IGZ_NAMESPACE_DOMAIN"] = igz_domain

    # workaround wrongly sqldb dsn in 2.8
    if (
        config.get("httpdb", {}).get("dsn")
        == "sqlite:///mlrun.sqlite3?check_same_thread=false"
    ):
        config["httpdb"]["dsn"] = "sqlite:////mlrun/db/mlrun.db?check_same_thread=false"

    # "disabled" is the helm chart default value, we don't want that value to be set cause when this value is set we
    # use it in calls to the Nuclio package, and when the Nuclio package receives a value it simply uses it, and
    # obviously "disabled" is not the right address.. when the Nuclio package doesn't receive a value it doing "best
    # effort" to try and determine the URL, we want this "best effort" so overriding the "disabled" value
    if config.get("nuclio_dashboard_url") == "disabled":
        config["nuclio_dashboard_url"] = ""

    if uisvc and not config.get("ui_url"):
        if igz_domain:
            config["ui_url"] = f"https://mlrun-ui.{igz_domain}"

    if config.get("log_level"):
        import mlrun.utils.logger

        # logger created (because of imports mess) before the config is loaded (in tests), therefore we're changing its
        # level manually
        mlrun.utils.logger.set_logger_level(config["log_level"])
    # The default function pod resource values are of type str; however, when reading from environment variable numbers,
    # it converts them to type int if contains only number, so we want to convert them to str.
    _convert_resources_to_str(config)
    return config


# populate config, skip errors when setting the config attributes and issue warnings instead
# this is to avoid failure when doing `import mlrun` and the dbpath (API service) is incorrect or down
_populate(skip_errors=True)
