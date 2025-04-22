# Copyright 2025 Iguazio
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

import mlrun_pipelines.client
import mlrun_pipelines.utils


@pytest.fixture
def client(monkeypatch):
    client_klass = mlrun_pipelines.client.Client
    client_klass.get_kfp_healthz = unittest.mock.MagicMock()
    monkeypatch.setattr("kubernetes.config.load_incluster_config", lambda: None)
    return client_klass()


@pytest.mark.parametrize(
    "original_name, project, expected",
    [
        ("sample_run", "projectX", "projectX-Retry of sample_run"),
        ("projectX-sample_run", "projectX", "projectX-Retry of sample_run"),
        ("Retry of sample_run", "projectX", "projectX-Retry of sample_run"),
        ("projectX-Retry of sample_run", "projectX", "projectX-Retry of sample_run"),
        (
            "  projectX-Retry of   sample_run  ",
            "projectX",
            "projectX-Retry of sample_run",
        ),
    ],
)
def test_normalize_retry_run(client, original_name, project, expected):
    assert client._normalize_retry_run(original_name, project) == expected
