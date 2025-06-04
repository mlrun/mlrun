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


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("simple", "simple"),  # lowercase, already valid
        ("MiXeD_CaSe", "mixed_case"),  # mixed case → lower-case kept
        ("with space", "with_space"),  # spaces → underscore
        ("double  space", "double_space"),  # condensed invalid runs → single _
        ("leading-", "leading"),  # leading invalid char stripped
        ("trailing!", "trailing"),  # trailing invalid char stripped
        ("--many!!bad$$chars--", "many_bad_chars"),  # multiple invalid segments
        ("___already_ok___", "already_ok"),  # leading/trailing underscores removed
        ("", ""),  # empty string stays empty
    ],
)
def test_sanitize_expected(raw, expected):
    assert mlrun_pipelines.client.sanitize_input_name(raw) == expected


def test_idempotent():
    """Calling the function twice should be a no-op."""
    name = "a__valid_name"
    assert (
        mlrun_pipelines.client.sanitize_input_name(
            mlrun_pipelines.client.sanitize_input_name(name)
        )
        == name
    )
