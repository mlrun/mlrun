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

import mlrun_pipelines.ops as ops


class _FakeTask:
    def __init__(self):
        self.env_vars = {}

    def set_env_variable(self, name, value):
        self.env_vars[name] = value


@pytest.mark.parametrize("kind", ["job", "spark", "mpijob"])
def test_add_default_env_sets_mlrun_runtime_kind(kind):
    """Regression test for ML-13046: KFP step pods never got MLRUN_RUNTIME_KIND."""
    task = _FakeTask()
    function = unittest.mock.MagicMock(kind=kind)

    ops.add_default_env(task, function)

    assert task.env_vars["MLRUN_RUNTIME_KIND"] == kind
