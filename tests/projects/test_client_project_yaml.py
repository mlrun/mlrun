# Copyright 2026 Iguazio
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

"""B: project.yaml opt-out under ``client.session()``."""

from __future__ import annotations

import pytest

import mlrun
from mlrun import Client, Credentials
from tests.common_fixtures import RunDBMock


@pytest.fixture
def _mock_dbpath(monkeypatch):
    monkeypatch.setattr(mlrun.mlconf, "dbpath", "https://mock-server")


def test_get_or_create_project_skips_disk_in_client_session(tmp_path, _mock_dbpath):
    """Inside ``client.session()``, ``get_or_create_project`` neither
    reads nor writes ``project.yaml`` on disk.

    Pre-condition: a ``project.yaml`` with distinctive content already
    sits in the context dir. Outside a session, today's behavior would
    (a) load that yaml when the requested project isn't in the DB and
    (b) overwrite it via ``project.save()`` once the new project is
    constructed.
    """
    disk_yaml = tmp_path / "project.yaml"
    disk_yaml.write_text(
        "kind: project\n"
        "metadata:\n"
        "  name: disk-project\n"
        "spec:\n"
        "  description: from-disk-do-not-load\n"
    )
    disk_yaml_mtime = disk_yaml.stat().st_mtime
    files_before = sorted(p.name for p in tmp_path.iterdir())

    client = Client(credentials=Credentials(token="t"))
    client._http_db = RunDBMock()

    with client.session():
        project = mlrun.get_or_create_project(
            "other-project",
            context=str(tmp_path),
            allow_cross_project=True,
        )

    # No disk write: tmp_path file set unchanged, existing yaml untouched.
    files_after = sorted(p.name for p in tmp_path.iterdir())
    assert files_after == files_before
    assert disk_yaml.stat().st_mtime == disk_yaml_mtime

    # No disk read: the on-disk yaml's content did not leak into the project.
    assert project.spec.description != "from-disk-do-not-load"
    assert project.metadata.name == "other-project"
