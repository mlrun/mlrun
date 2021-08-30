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

import copy
import json
import os
import threading
import yaml

from .default import default_config

env_prefix = "MLRUN_"
env_file_key = f"{env_prefix}CONFIG_FILE"


class ConfigLoader:
    loaded = False
    _load_lock = threading.Lock()
    _config_dict = {}

    @staticmethod
    def load_config():
        if ConfigLoader.loaded:
            return ConfigLoader._config_dict

        return ConfigLoader.reload_config()

    @staticmethod
    def reload_config():
        with ConfigLoader._load_lock:
            ConfigLoader.loaded = True
            ConfigLoader._config_dict = ConfigLoader._load_config()
            return ConfigLoader._config_dict

    @staticmethod
    def _load_config(env=None):
        config = copy.deepcopy(default_config)

        config_path = os.environ.get(env_file_key)
        if config_path:
            with open(config_path) as fp:
                data = yaml.safe_load(fp)

            if not isinstance(data, dict):
                raise TypeError(f"configuration in {config_path} not a dict")

            ConfigLoader._deep_merge(data, config)

        data = ConfigLoader._read_env(env)
        if data:
            ConfigLoader._deep_merge(data, config)

        return config

    @staticmethod
    def _read_env(env=None, prefix=env_prefix):
        """Read configuration from environment"""
        env = os.environ if env is None else env

        config = {}
        for key, value in env.items():
            if not key.startswith(prefix) or key == env_file_key:
                continue
            try:
                value = json.loads(value)  # values can be JSON encoded
            except ValueError:
                pass  # Leave as string
            key = key[len(prefix) :]  # Trim MLRUN_
            path = key.lower().split("__")  # 'A__B' → ['a', 'b']
            cfg = config
            while len(path) > 1:
                name, *path = path
                cfg = cfg.setdefault(name, {})
            cfg[path[0]] = value

        # It's already a standard to set this env var to configure the v3io api, so we're supporting it (instead
        # of MLRUN_V3IO_API)
        v3io_api = env.get("V3IO_API")
        if v3io_api:
            config["v3io_api"] = v3io_api

        # It's already a standard to set this env var to configure the v3io framesd, so we're supporting it (instead
        # of MLRUN_V3IO_FRAMESD)
        v3io_framesd = env.get("V3IO_FRAMESD")
        if v3io_framesd:
            config["v3io_framesd"] = v3io_framesd

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

        return config

    @staticmethod
    def _deep_merge(source, destination):
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                ConfigLoader._deep_merge(value, node)
            else:
                destination[key] = value

        return destination
