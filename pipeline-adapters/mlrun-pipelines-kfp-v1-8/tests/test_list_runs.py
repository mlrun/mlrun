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

import mlrun.utils
import mlrun_pipelines.client


@pytest.fixture
def client(monkeypatch):
    client_klass = mlrun_pipelines.client.Client
    client_klass.get_kfp_healthz = unittest.mock.MagicMock()
    monkeypatch.setattr("kubernetes.config.load_incluster_config", lambda: None)
    monkeypatch.setattr(client_klass, "_determine_server_major_version", lambda self: 2)
    return client_klass(logger=mlrun.utils.logger)


def test_list_runs_no_matching_experiments_does_not_scan_all_runs(client, monkeypatch):
    """
    On KFP v2, a project that resolves to zero matching experiments must not fall
    through to an unscoped cluster-wide scan. Pipeline runs always live under a
    project-prefixed experiment, so no matching experiment means no runs for the
    project; the unscoped fallback times out callers such as project deletion.
    """
    monkeypatch.setattr(
        client,
        "_get_candidate_experiments_for_projects",
        lambda project_names, **_: [],
    )

    paginate_runs_mock = unittest.mock.MagicMock(
        return_value=iter([(["run-from-another-project"], None)])
    )
    list_runs_page_mock = unittest.mock.MagicMock(
        return_value=(["run-from-another-project"], None)
    )
    monkeypatch.setattr(client, "_paginate_runs", paginate_runs_mock)
    monkeypatch.setattr(client, "_list_runs", list_runs_page_mock)

    result = list(client.list_runs(project="empty-project"))

    assert result == []
    paginate_runs_mock.assert_not_called()
    list_runs_page_mock.assert_not_called()


def test_list_runs_wildcard_project_does_not_short_circuit(client, monkeypatch):
    """
    project="*" must keep running the intended unscoped scan: the experiment
    lookup is bypassed entirely so the empty-experiment guard cannot observe an
    "empty" result. Without this, the periodic project-summaries counter and
    tenant-wide pipelines listing would silently return nothing.
    """
    get_experiments_mock = unittest.mock.MagicMock(return_value=[])
    monkeypatch.setattr(
        client, "_get_candidate_experiments_for_projects", get_experiments_mock
    )
    paginate_runs_mock = unittest.mock.MagicMock(return_value=iter([([], None)]))
    monkeypatch.setattr(client, "_paginate_runs", paginate_runs_mock)

    list(client.list_runs(project="*"))

    get_experiments_mock.assert_not_called()
    paginate_runs_mock.assert_called_once()
