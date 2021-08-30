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

import base64
import binascii
import json
import urllib.parse

from .base import ConfigBase
from .loader import ConfigLoader


class Config(ConfigBase):

    dynamic_attributes = [
        "dbpath",
        "dask_kfp_image",
        "iguazio_api_url",
        "kfp_image",
    ]

    def __init__(self, cfg=None):
        super().__init__(cfg)

        # HACK to enable config property to both have dynamic default and to use the value from dict/env like other
        # configurations - we just need a key in the dict that is different than the property name, so simply adding
        # prefix underscore
        for attribute in self.dynamic_attributes:
            if attribute in self._cfg:
                self._cfg[f"_{attribute}"] = self._cfg[attribute]
                self._cfg.pop(attribute, None)

    def get_build_args(self):
        build_args = {}
        if self.httpdb.builder.build_args:
            build_args_json = base64.b64decode(
                self.httpdb.builder.build_args
            ).decode()
            build_args = json.loads(build_args_json)

        return build_args

    def get_default_function_node_selector(self):
        default_function_node_selector = {}
        if self.default_function_node_selector:
            default_function_node_selector_json_string = base64.b64decode(
                self.default_function_node_selector
            ).decode()
            default_function_node_selector = json.loads(
                default_function_node_selector_json_string
            )

        return default_function_node_selector

    @staticmethod
    def get_valid_function_priority_class_names():
        if not config.valid_function_priority_class_names:
            return []
        return list(set(config.valid_function_priority_class_names.split(",")))

    def get_storage_auto_mount_params(self):
        auto_mount_params = {}
        if self.storage.auto_mount_params:
            try:
                auto_mount_params = base64.b64decode(
                    self.storage.auto_mount_params, validate=True
                ).decode()
                auto_mount_params = json.loads(auto_mount_params)
            except binascii.Error:
                # Importing here to avoid circular dependencies
                from mlrun.utils import list2dict

                # String wasn't base64 encoded. Parse it using a 'p1=v1,p2=v2' format.
                mount_params = self.storage.auto_mount_params.split(",")
                auto_mount_params = list2dict(mount_params)
        if not isinstance(auto_mount_params, dict):
            raise TypeError(
                f"data in storage.auto_mount_params does not resolve to a dictionary: {auto_mount_params}"
            )

        return auto_mount_params

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

    @kfp_image.setter
    def kfp_image(self, value):
        self._kfp_image = value

    @property
    def dask_kfp_image(self):
        """
        See kfp_image property docstring for why we're defining this property
        """
        if not self._dask_kfp_image:
            # importing here to avoid circular dependency
            import mlrun.utils.helpers

            return mlrun.utils.helpers.enrich_image_url("mlrun/ml-base")
        return self._dask_kfp_image

    @dask_kfp_image.setter
    def dask_kfp_image(self, value):
        self._dask_kfp_image = value

    def resolve_ui_url(self):
        # ui_url is deprecated in favor of the ui.url (we created the ui block)
        # since the config class is used in a "recursive" way, we can't use property like we used in other places
        # since the property will need to be url, which exists in other structs as well
        return self.ui.url or self.ui_url

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

    @property
    def iguazio_api_url(self):
        """
        we want to be able to run with old versions of the service who runs the API (which doesn't configure this
        value) so we're doing best effort to try and resolve it from other configurations
        TODO: Remove this hack when 0.6.x is old enough
        """
        if not self._iguazio_api_url:
            if self.httpdb.builder.docker_registry and self.igz_version:
                return self._extract_iguazio_api_from_docker_registry_url()
        return self._iguazio_api_url

    def _extract_iguazio_api_from_docker_registry_url(self):
        docker_registry_url = self.httpdb.builder.docker_registry
        # add schema otherwise parsing go wrong
        if "://" not in docker_registry_url:
            docker_registry_url = f"http://{docker_registry_url}"
        parsed_registry_url = urllib.parse.urlparse(docker_registry_url)
        registry_hostname = parsed_registry_url.hostname
        # replace the first domain section (app service name) with dashboard
        first_dot_index = registry_hostname.find(".")
        if first_dot_index < 0:
            # if not found it's not the format we know - can't resolve the api url from the registry url
            return ""
        return f"https://dashboard{registry_hostname[first_dot_index:]}"

    @iguazio_api_url.setter
    def iguazio_api_url(self, value):
        self._iguazio_api_url = value


config = Config(ConfigLoader.load_config())
