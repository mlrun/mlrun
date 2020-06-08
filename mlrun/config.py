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

import json
import os
from collections.abc import Mapping
from distutils.util import strtobool
from os.path import expanduser
from threading import Lock
from urllib.parse import urlparse
from . import __version__

import yaml

env_prefix = 'MLRUN_'
env_file_key = '{}CONIFG_FILE'.format(env_prefix)
_load_lock = Lock()
_none_type = type(None)


default_config = {
    'namespace': 'default-tenant',   # default kubernetes namespace
    'dbpath': '',                    # db/api url
    # url to nuclio dashboard api (can be with user & token, e.g. https://username:password@dashboard-url.com)
    'nuclio_dashboard_url': '',
    'ui_url': '',                    # remote/external mlrun UI url (for hyperlinks)
    'remote_host': '',
    'version': '',                   # will be set to current version
    'images_tag': '',                # tag to use with mlrun images e.g. mlrun/mlrun (defaults to version)
    'kfp_ttl': '86400',              # KFP ttl in sec, after that completed PODs will be deleted
    'kfp_image': '',                 # image to use for KFP runner (defaults to mlrun/mlrun)
    'kaniko_version': 'v0.19.0',     # kaniko builder version
    'package_path': 'mlrun',         # mlrun pip package
    'default_image': 'python:3.6-jessie',
    'default_project': 'default',    # default project name
    'default_archive': '',           # default remote archive URL (for build tar.gz)
    'mpijob_crd_version': '',        # mpijob crd version (e.g: "v1alpha1". must be in: mlrun.runtime.MPIJobCRDVersions)
    'hub_url': 'https://raw.githubusercontent.com/mlrun/functions/{tag}/{name}/function.yaml',
    'ipython_widget': True,
    'log_level': 'ERROR',
    'submit_timeout': '180',         # timeout when submitting a new k8s resource
    'artifact_path': '',             # default artifacts path/url
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


class Config:
    _missing = object()

    def __init__(self, cfg=None):
        cfg = {} if cfg is None else cfg
        # Can't use self._cfg = cfg → infinite recursion
        object.__setattr__(self, '_cfg', cfg)

    def __getattr__(self, attr):
        val = self._cfg.get(attr, self._missing)
        if val is self._missing:
            raise AttributeError(attr)

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

    def update(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)

    def dump_yaml(self, stream=None):
        return yaml.dump(self._cfg, stream, default_flow_style=False)

    @staticmethod
    def reload():
        _populate()


# Global configuration
config = Config(default_config)


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

    config = Config(default_config)
    config_path = os.environ.get(env_file_key)
    if config_path:
        with open(config_path) as fp:
            data = yaml.safe_load(fp)

        if not isinstance(data, dict):
            raise TypeError(f'configuration in {config_path} not a dict')

        config.update(data)

    data = read_env(env)
    if data:
        config.update(data)


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
        key = key[len(env_prefix):]  # Trim MLRUN_
        path = key.lower().split('__')  # 'A__B' → ['a', 'b']
        cfg = config
        while len(path) > 1:
            name, *path = path
            cfg = cfg.setdefault(name, {})
        cfg[path[0]] = value

    # check for mlrun-api or db kubernetes service
    svc = env.get('MLRUN_API_PORT')
    if svc and not config.get('dbpath'):
        config['dbpath'] = 'http://mlrun-api:{}'.format(
            default_config['httpdb']['port'] or 8080)

    uisvc = env.get('MLRUN_UI_SERVICE_HOST')
    igz_domain = env.get('IGZ_NAMESPACE_DOMAIN')

    # workaround to try and detect IGZ domain in 2.8
    if not igz_domain and 'DEFAULT_DOCKER_REGISTRY' in env:
        registry = env['DEFAULT_DOCKER_REGISTRY']
        if registry.startswith('docker-registry.default-tenant'):
            igz_domain = registry[len('docker-registry.'):]
            if ':' in igz_domain:
                igz_domain = igz_domain[:igz_domain.rfind(':')]
            env['IGZ_NAMESPACE_DOMAIN'] = igz_domain

    if uisvc and not config.get('ui_url'):
        if igz_domain:
            config['ui_url'] = 'https://mlrun-ui.{}'.format(igz_domain)

    if not config.get('kfp_image'):
        tag = __version__ or 'latest'
        config['kfp_image'] = 'mlrun/mlrun:{}'.format(tag)

    config['version'] = __version__

    return config


_populate()
