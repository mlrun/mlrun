# Copyright 2023 Iguazio
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
#
import os.path
import pathlib
import typing
import unittest.mock

import alembic
import alembic.config
import pytest

import server.api.utils.db.alembic
from mlrun import mlconf


class Constants:
    revision_history = ["revision2", "revision1"]
    initial_revision = "revision1"
    latest_revision = "revision2"
    unknown_revision = "revision3"


def test_no_database_exists(mock_alembic, mock_database):
    mock_database(db_file_exists=False)
    alembic_util = server.api.utils.db.alembic.AlembicUtil(pathlib.Path(""))
    alembic_util.init_alembic()
    assert mock_alembic.stamp_calls == []
    assert mock_alembic.upgrade_calls == ["head"]


def test_database_exists_no_revision(mock_alembic, mock_database):
    mock_database()
    alembic_util = server.api.utils.db.alembic.AlembicUtil(pathlib.Path(""))
    alembic_util.init_alembic()

    assert mock_alembic.upgrade_calls == ["head"]


def test_database_exists_known_revision(mock_alembic, mock_database):
    mock_database(current_revision=Constants.initial_revision)
    alembic_util = server.api.utils.db.alembic.AlembicUtil(pathlib.Path(""))
    alembic_util.init_alembic()
    assert mock_alembic.stamp_calls == []
    assert mock_alembic.upgrade_calls == ["head"]


def test_database_exists_unknown(mock_alembic, mock_database):
    mock_database(current_revision=Constants.unknown_revision)
    alembic_util = server.api.utils.db.alembic.AlembicUtil(pathlib.Path(""))
    alembic_util.init_alembic()
    assert mock_alembic.stamp_calls == []
    assert mock_alembic.upgrade_calls == ["head"]


@pytest.fixture()
def mock_database(
    monkeypatch, mock_alembic, mock_db_file_name
) -> typing.Callable[[list[str], str, bool, bool], None]:
    def _mock_database(
        revision_history: list[str] = None,
        current_revision: str = "",
        db_file_exists: bool = True,
        db_backup_exists: bool = True,
    ):
        revision_history = revision_history or Constants.revision_history

        def _db_file_exists(file_name: str) -> bool:
            if file_name == mock_db_file_name:
                return db_file_exists
            else:
                return db_backup_exists

        monkeypatch.setattr(os.path, "isfile", _db_file_exists)

        def _current_revision(alembic_config: typing.Any):
            if current_revision != "" and current_revision not in revision_history:
                raise Exception(
                    f"Can't locate revision identified by '{current_revision}'"
                )

            alembic_config.print_stdout(current_revision)

        mock_alembic.current = _current_revision

        def _revision_history(alembic_config: typing.Any):
            for revision in revision_history:
                alembic_config.print_stdout(f"none -> {revision}, revision name")

        mock_alembic.history = _revision_history

    return _mock_database


@pytest.fixture()
def mock_db_file_name(monkeypatch) -> str:
    db_file_name = "test.db"
    monkeypatch.setattr(mlconf.httpdb, "dsn", db_file_name)
    return db_file_name


class MockAlembicCommand:
    def __init__(self):
        self.stamp_calls = []
        self.upgrade_calls = []

    def stamp(self, alembic_config: typing.Any, revision: str):
        self.stamp_calls.append(revision)

    def upgrade(self, alembic_config: typing.Any, revision: str):
        self.upgrade_calls.append(revision)


@pytest.fixture()
def mock_alembic(monkeypatch) -> MockAlembicCommand:
    mocked_alembic_command = MockAlembicCommand()
    monkeypatch.setattr(alembic, "command", mocked_alembic_command)
    monkeypatch.setattr(alembic.config, "Config", unittest.mock.Mock())
    return mocked_alembic_command
