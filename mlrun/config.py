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
    'namespace': 'default-tenant',
    'dbpath': '',
    'ui_url': '',
    'remote_host': '',
    'kfp_image': '',
    'kaniko_version': 'v0.14.0',
    'package_path': 'mlrun',
    'default_image': 'python:3.6-jessie',
    'default_project': 'default',
    'default_archive': '',
    'ipython_widget': True,
    'log_level': 'ERROR',
    'k8s_submit_timeout': '120',
    'httpdb': {
        'port': 8080,
        'dirpath': expanduser('~/.mlrun/db'),
        'dsn': 'sqlite:///:memory:?check_same_thread=false',
        'debug': False,
        'user': '',
        'password': '',
        'token': '',
        'logs_path': expanduser('~/.mlrun/logs'),
        'files_path': '',
        'db_type': 'filerundb',
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
    svc = env.get('MLRUN_API_PORT', env.get('MLRUN_DB_PORT'))
    if svc and not config.get('dbpath'):
        config['dbpath'] = 'http://' + urlparse(svc).netloc
    if not config.get('kfp_image'):
        tag = __version__ or 'latest'
        config['kfp_image'] = 'mlrun/mlrun:{}'.format(tag)

    return config


_populate()
