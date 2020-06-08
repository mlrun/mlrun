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

from copy import deepcopy
from typing import IO
from mlrun import config as mlconf
from contextlib import contextmanager
from os import environ
import yaml
from tempfile import NamedTemporaryFile

import pytest

ns_env_key = f'{mlconf.env_prefix}NAMESPACE'


@pytest.fixture
def config():
    yield mlconf.Config(deepcopy(mlconf.default_config))


@pytest.fixture
def create_yaml_config():
    temp_file: IO = None

    def _create_file(**kw):
        nonlocal temp_file
        temp_file = NamedTemporaryFile(mode='wt', suffix='.yml', delete=False)
        yaml.dump(kw, temp_file, default_flow_style=False)
        temp_file.flush()
        return temp_file.name

    try:
        yield _create_file
    finally:
        if temp_file:
            temp_file.close()


def test_sanity(config):
    assert config._cfg is not mlconf.config._cfg


def test_nothing(config):
    expected = mlconf.default_config['namespace']
    assert config.namespace == expected, 'namespace changed'


def test_file(config, create_yaml_config):
    ns = 'banana'
    config_path = create_yaml_config(namespace=ns)
    with _patch_env({mlconf.env_file_key: config_path}):
        config = config.load()

    assert config.namespace == ns, 'not populated from file'


def test_env(config):
    ns = 'orange'
    with _patch_env({ns_env_key: ns}):
        config = config.load()

    assert config.namespace == ns, 'not populated from env'


def test_env_override(config, create_yaml_config):
    env_ns = 'daffy'
    config_ns = 'bugs'

    config_path = create_yaml_config(namespace=config_ns)
    env = {
        mlconf.env_file_key: config_path,
        ns_env_key: env_ns,
    }

    with _patch_env(env):
        config = config.load()

    assert config.namespace == env_ns, 'env did not override'


def test_can_set(config):
    config._cfg['x'] = {'y': 10}
    val = 90
    config.x.y = val
    assert config.x.y == val, 'bad config update'


def test_update(config):
    config._cfg['x'] = None
    config._cfg['y'] = None

    config.update(x=1)
    assert config.x == 1, 'failed to update'
    config.update({'y': 2})
    assert config.y == 2, 'failed to update'


@contextmanager
def _patch_env(kw):

    # create a backup
    os_environ_copy = environ.copy()
    try:

        # update environ with given kw
        environ.update(kw)
        yield
    finally:

        # removes given kw from environ
        for key in kw.keys():
            del environ[key]

        # redo environ from backup
        environ.update(os_environ_copy)
