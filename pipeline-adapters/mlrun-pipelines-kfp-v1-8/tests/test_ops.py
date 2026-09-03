# Copyright 2026 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest.mock

import pytest
from kubernetes import client as k8s_client

import mlrun_pipelines.ops as ops


class _FakeContainer:
    def __init__(self):
        self.env_vars = []

    def add_env_variable(self, env_var):
        self.env_vars.append(env_var)


class _FakeContainerOp:
    def __init__(self):
        self.container = _FakeContainer()


def _env_value(cop, name):
    for env_var in cop.container.env_vars:
        if env_var.name == name:
            return env_var.value
    raise AssertionError(f"{name} env var was not set on the container")


@pytest.mark.parametrize("kind", ["job", "spark", "mpijob"])
def test_add_default_env_sets_mlrun_runtime_kind(kind):
    """Regression test for ML-13046: KFP step pods never got MLRUN_RUNTIME_KIND."""
    cop = _FakeContainerOp()
    function = unittest.mock.MagicMock(kind=kind)

    ops.add_default_env(k8s_client, cop, function)

    assert _env_value(cop, "MLRUN_RUNTIME_KIND") == kind
