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
"""
Configuration system.

Configuration can be in either a configuration file specified by
MLRUN_CONFIG_FILE environment variable or by environmenet variables.

Environment variables are in the format "MLRUN_httpdb__port=8080". This will be
mapped to config.httpdb.port. Values should be in JSON format.
"""

import copy
import json
import os
from collections.abc import Mapping
from distutils.util import strtobool
from os.path import expanduser
from threading import Lock

import yaml

env_prefix = "MLRUN_"
env_file_key = "{}CONIFG_FILE".format(env_prefix)
_load_lock = Lock()
_none_type = type(None)


default_config = {
    "namespace": "",  # default kubernetes namespace
    "dbpath": "",  # db/api url
    # url to nuclio dashboard api (can be with user & token, e.g. https://username:password@dashboard-url.com)
    "nuclio_dashboard_url": "",
    "nest_asyncio_enabled": "",  # enable import of nest_asyncio for corner cases with old jupyter, set "1"
    "ui_url": "",  # remote/external mlrun UI url (for hyperlinks) (This is deprecated in favor of the ui block)
    "remote_host": "",
    "version": "",  # will be set to current version
    "images_tag": "",  # tag to use with mlrun images e.g. mlrun/mlrun (defaults to version)
    "images_registry": "",  # registry to use with mlrun images e.g. quay.io/ (defaults to empty, for dockerhub)
    "kfp_ttl": "14400",  # KFP ttl in sec, after that completed PODs will be deleted
    "kfp_image": "",  # image to use for KFP runner (defaults to mlrun/mlrun)
    "igz_version": "",  # the version of the iguazio system the API is running on
    "spark_app_image": "",  # image to use for spark operator app runtime
    "spark_app_image_tag": "",  # image tag to use for spark opeartor app runtime
    "builder_alpine_image": "alpine:3.13.1",  # builder alpine image (as kaniko's initContainer)
    "package_path": "mlrun",  # mlrun pip package
    "default_image": "python:3.6-jessie",
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
    "runs_monitoring_interval": "5",
    # the grace period (in seconds) that will be given to runtime resources (after they're in terminal state)
    # before deleting them
    "runtime_resources_deletion_grace_period": "14400",
    # sets the background color that is used in printed tables in jupyter
    "background_color": "#4EC64B",
    "artifact_path": "",  # default artifacts path/url
    # url template for default model tracking stream
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
        "scheduling": {
            # the minimum interval that will be allowed between two scheduled jobs - e.g. a job wouldn't be
            # allowed to be scheduled to run more then 2 times in X. Can't be less then 1 minute
            "min_allowed_interval": "10 minutes"
        },
        "projects": {
            "leader": "mlrun",
            "followers": "",
            "periodic_sync_interval": "1 minute",
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
        },
    },
    "model_endpoint_monitoring": {
        "container": "projects",
        "stream_url": "v3io:///projects/{project}/model-endpoints/stream",
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
    },
    "feature_store": {
        "data_prefixes": {
            "default": "v3io:///projects/{project}/fs/{kind}",
            "nosql": "v3io:///projects/{project}/fs/{kind}",
        },
        "default_targets": "parquet,nosql",
    },
    "ui": {
        "projects_prefix": "projects",  # The UI link prefix for projects
        "url": "",  # remote/external mlrun UI url (for hyperlinks)
    },
}


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

    def update(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)

    def dump_yaml(self, stream=None):
        return yaml.dump(self._cfg, stream, default_flow_style=False)

    @classmethod
    def from_dict(cls, dict_):
        return cls(copy.deepcopy(dict_))

    def to_dict(self):
        return copy.copy(self._cfg)

    @staticmethod
    def reload():
        _populate()

    @property
    def version(self):
        # importing here to avoid circular dependency
        from mlrun.utils.version import Version

        return Version().get()["version"]

    @property
    def kfp_image(self):
        """
        When this configuration is not set we want to set it to mlrun/mlrun, but we need to use the enrich_image method.
        The problem is that the mlrun.utils.helpers module is importing the config (this) module, so we must import the
        module inside this function (and not on initialization), and then calculate this property value here.
        """
        if not self._kfp_image:
            # importing here to avoid circular dependency
            import mlrun.utils.helpers

            return mlrun.utils.helpers.enrich_image_url("mlrun/mlrun")
        return self._kfp_image

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
            mlrun.db.get_run_db(value)


# Global configuration
config = Config.from_dict(default_config)


def _populate():
    """Populate configuration from config file (if exists in environment) and
    from environment variables.

    populate will run only once, after first call it does nothing.
    """
    global _loaded

    with _load_lock:
        _do_populate()


def _do_populate(env=None):
    global config

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

        config.update(data)

    data = read_env(env)
    if data:
        config.update(data)

    # HACK to enable kfp_image property to both have dynamic default and to use the value from dict/env like
    # other configurations
    config._cfg["_kfp_image"] = config._cfg["kfp_image"]
    del config._cfg["kfp_image"]


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

    # TODO: remove this - and verify dbpath is set correctly in all flows
    # Here we're just guessing that there is a service named mlrun-api, if there is, there will be an env var for it's
    # port - MLRUN_API_PORT - so we're using the env var existence to know whether our guess is right.
    # the existence of config.httpdb.api_url tell that we're running in an API context so no need to set the dbpath
    svc = env.get("MLRUN_API_PORT")
    if svc and not config.get("dbpath") and not config.get("httpdb", {}).get("api_url"):
        config["dbpath"] = "http://mlrun-api:{}".format(
            default_config["httpdb"]["port"] or 8080
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
            config["ui_url"] = "https://mlrun-ui.{}".format(igz_domain)

    if config.get("log_level"):
        import mlrun.utils.logger

        # logger created (because of imports mess) before the config is loaded (in tests), therefore we're changing its
        # level manually
        mlrun.utils.logger.set_logger_level(config["log_level"])

    return config


_populate()
