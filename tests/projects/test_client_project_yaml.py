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

"""project.yaml opt-out under ``client.session()``"""

from __future__ import annotations

import pytest

import mlrun
import mlrun.errors
import mlrun.projects.project
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


def test_load_project_rejects_yaml_url_in_client_session(tmp_path, _mock_dbpath):
    yaml_path = tmp_path / "project.yaml"
    yaml_path.write_text("kind: project\nmetadata:\n  name: x\n")

    client = Client(credentials=Credentials(token="t"))
    with client.session(), pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.load_project(context=str(tmp_path), url=str(yaml_path))


@pytest.mark.parametrize(
    "url,patched_fn",
    [
        ("git://github.com/foo/bar", "clone_git"),
        ("https://example.com/foo.tar.gz", "clone_tgz"),
        ("https://example.com/foo.zip", "clone_zip"),
    ],
)
def test_load_project_rejects_remote_url_without_fetch(
    tmp_path, monkeypatch, _mock_dbpath, url, patched_fn
):
    """git/tgz/zip URLs must fast-fail inside ``client.session()`` — no clone
    or extract should happen before the gate fires.
    """
    fetch_calls: list = []

    def _record(*args, **kwargs):
        fetch_calls.append((args, kwargs))
        return (args[0], None) if patched_fn == "clone_git" else None

    monkeypatch.setattr(f"mlrun.projects.project.{patched_fn}", _record)

    client = Client(credentials=Credentials(token="t"))
    with client.session(), pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.load_project(context=str(tmp_path), url=url)

    assert fetch_calls == [], (
        f"{patched_fn} was invoked before the client-session gate fired"
    )


def test_load_project_no_url_unresolvable_in_client_session(tmp_path, _mock_dbpath):
    """No-url ``load_project`` inside a session must raise
    ``MLRunInvalidArgumentError`` (not ``MLRunNotFoundError``) — the DB was
    never queried, so a 'not found in DB' error misleads.
    """
    client = Client(credentials=Credentials(token="t"))
    client._http_db = RunDBMock()
    with client.session(), pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.load_project(context=str(tmp_path), name="no-such-project")


def test_user_project_rejected_in_client_session(_mock_dbpath):
    client = Client(credentials=Credentials(token="t"))
    with client.session(), pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.get_or_create_project("base", user_project=True)


def test_resolve_artifact_owner_ignores_env_in_client_session(
    monkeypatch, _mock_dbpath
):
    """``_resolve_artifact_owner`` must skip ``V3IO_USERNAME`` inside a session;
    outside, the env fallback still applies.
    """
    monkeypatch.setenv("V3IO_USERNAME", "process-user")
    project = mlrun.projects.project.MlrunProject.from_dict(
        {"metadata": {"name": "owner-test"}}
    )

    client = Client(credentials=Credentials(token="t"))
    with client.session():
        assert project._resolve_artifact_owner() is None

    assert project._resolve_artifact_owner() == "process-user"
