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
MLRUN_CONFIG_FILE environment variable or by environment variables.

Environment variables are in the format "MLRUN_httpdb__port=8080". This will be
mapped to config.httpdb.port. Values should be in JSON format.
"""

import json
import os
from collections.abc import Mapping
from os.path import expanduser
from . import __version__

import yaml

env_prefix = 'MLRUN_'
env_file_key = f'{env_prefix}CONIFG_FILE'

default_config = {

    # default kubernetes namespace
    'namespace': 'default-tenant',

    # db/api url
    'dbpath': '',

    # url to nuclio dashboard api (can be with user & token, e.g. https://username:password@dashboard-url.com)
    'nuclio_dashboard_url': '',

    # remote/external mlrun UI url (for hyperlinks)
    'ui_url': '',
    'remote_host': '',

    # will be set to current version
    'version': '',

    # tag to use with mlrun images e.g. mlrun/mlrun (defaults to version)
    'images_tag': '',

    # KFP ttl in sec, after that completed PODs will be deleted
    'kfp_ttl': '86400',

    # image to use for KFP runner (defaults to mlrun/mlrun)
    'kfp_image': '',

    # kaniko builder version
    'kaniko_version': 'v0.19.0',

    # mlrun pip package
    'package_path': 'mlrun',
    'default_image': 'python:3.6-jessie',

    # default project name
    'default_project': 'default',

    # default remote archive URL (for build tar.gz)
    'default_archive': '',

    # mpijob crd version (e.g: "v1alpha1". must be in: mlrun.runtime.MPIJobCRDVersions)
    'mpijob_crd_version': '',
    'hub_url': 'https://raw.githubusercontent.com/mlrun/functions/{tag}/{name}/function.yaml',
    'ipython_widget': True,
    'log_level': 'ERROR',

    # timeout when submitting a new k8s resource
    'submit_timeout': '180',

    # default artifacts path/url
    'artifact_path': '',
    'httpdb': {
        'port': 8080,
        'dirpath': expanduser('~/.mlrun/db'),
        'dsn': 'sqlite:////tmp/mlrun.db?check_same_thread=false',
        'debug': False,
        'user': '',
        'password': '',
        'token': '',
        'logs_path': expanduser('~/.mlrun/logs'),
        'data_volume': '',
        'real_path': '',
        'db_type': 'sqldb',
    },
}


class Config(object):

    def __init__(self, cfg=None):
        cfg = {} if cfg is None else cfg

        # Can't use self._cfg = cfg → infinite recursion
        object.__setattr__(self, '_cfg', cfg)

    def __getattr__(self, attr):
        if attr not in self._cfg:
            raise AttributeError(attr)
        val = self._cfg[attr]
        if isinstance(val, Mapping):
            return self.__class__(val)
        return val

    def __setattr__(self, attr, value):
        self._cfg[attr] = value

    def __dir__(self):
        return list(self._cfg) + dir(self.__class__)

    def __repr__(self):
        name = self.__class__.__name__
        return f'{name}({self._cfg!r})'

    def update(self, cfg=None, **cfg_kwargs):
        """
        Update self from dict, followed by update by given kwargs
        """
        if isinstance(cfg, dict):
            self._update(cfg)
        self._update(cfg_kwargs)

    def dump_yaml(self, stream=None):
        return yaml.dump(self._cfg, stream, default_flow_style=False)

    @classmethod
    def load(cls, env=None):
        """
        reload configuration from config file (if exists in environment) and from environment variables.
        """
        env = env if env else os.environ
        config_instance = cls(default_config)
        config_path = env.get(env_file_key)
        if config_path:
            with open(config_path) as fp:
                data = yaml.safe_load(fp)

            if not isinstance(data, dict):
                raise TypeError(f'configuration in {config_path} not a dict')

            config_instance.update(data)

        config_instance._enrich_from_env(env)
        return config_instance

    def _has_attr(self, attr):
        return attr in self._cfg

    def _update(self, d: dict):
        for key, value in d.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)

    def _enrich_from_env(self, env, prefix=env_prefix):
        """Read configuration from environment"""

        for key, value in env.items():
            if not key.startswith(prefix) or key == env_file_key:
                continue
            try:

                # values can be JSON encoded
                value = json.loads(value)
            except ValueError:

                # Leave as string
                pass

            # trim prefix
            key = key[len(prefix):]

            # 'A__B' → ['a', 'b']
            path = key.lower().split('__')

            c = self
            while len(path) > 1:
                name, *path = path
                c._cfg.setdefault(name, {})
                c = getattr(self, name)
            setattr(c, path[0], value)

        # check for mlrun-api or db kubernetes service
        api_port = env.get('MLRUN_API_PORT')
        if api_port and not self._has_attr('dbpath'):
            self.dbpath = f'http://mlrun-api:{self.httpdb.port or 8080}'

        # workaround to try and detect IGZ domain in 2.8
        igz_domain = env.get('IGZ_NAMESPACE_DOMAIN')
        if not igz_domain and 'DEFAULT_DOCKER_REGISTRY' in env:
            registry = env['DEFAULT_DOCKER_REGISTRY']
            if registry.startswith('docker-registry.default-tenant'):
                igz_domain = registry[len('docker-registry.'):]
                if ':' in igz_domain:
                    igz_domain = igz_domain[:igz_domain.rfind(':')]
                env['IGZ_NAMESPACE_DOMAIN'] = igz_domain

        # set ui service url
        uisvc = env.get('MLRUN_UI_SERVICE_HOST')
        if igz_domain and uisvc and not self._has_attr('ui_url'):
            self.ui_url = f'https://mlrun-ui.{igz_domain}'

        # set image
        if not self._has_attr('kfp_image'):
            self.kfp_image = f'mlrun/mlrun:{__version__ or "latest"}'

        # set version
        self.version = __version__


config = Config.load()
