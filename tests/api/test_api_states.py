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
#
import http
import unittest.mock

import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.initial_data
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.db.alembic
import mlrun.api.utils.db.backup
import mlrun.api.utils.db.sqlite_migration


def test_offline_state(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.offline
    response = client.get("healthz")
    assert response.status_code == http.HTTPStatus.OK.value

    response = client.get("projects")
    assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
    assert "API is in offline state" in response.text


def test_migrations_states(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    expected_message_map = {
        mlrun.api.schemas.APIStates.waiting_for_migrations: "API is waiting for migrations to be triggered",
        mlrun.api.schemas.APIStates.migrations_in_progress: "Migrations are in progress",
        mlrun.api.schemas.APIStates.migrations_failed: "Migrations failed",
    }
    for state, expected_message in expected_message_map.items():
        mlrun.mlconf.httpdb.state = state
        response = client.get("healthz")
        assert response.status_code == http.HTTPStatus.OK.value

        response = client.get("projects/some-project/background-tasks/some-task")
        assert response.status_code == http.HTTPStatus.NOT_FOUND.value

        response = client.get("client-spec")
        assert response.status_code == http.HTTPStatus.OK.value

        response = client.get("projects")
        assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
        assert expected_message in response.text


def test_init_data_migration_required_recognition(monkeypatch) -> None:
    sqlite_migration_util_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        mlrun.api.utils.db.sqlite_migration,
        "SQLiteMigrationUtil",
        sqlite_migration_util_mock,
    )
    alembic_util_mock = unittest.mock.Mock()
    monkeypatch.setattr(mlrun.api.utils.db.alembic, "AlembicUtil", alembic_util_mock)
    is_latest_data_version_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        mlrun.api.initial_data, "_is_latest_data_version", is_latest_data_version_mock
    )
    db_backup_util_mock = unittest.mock.Mock()
    monkeypatch.setattr(mlrun.api.utils.db.backup, "DBBackupUtil", db_backup_util_mock)
    perform_schema_migrations_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        mlrun.api.initial_data,
        "_perform_schema_migrations",
        perform_schema_migrations_mock,
    )
    perform_database_migration_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        mlrun.api.initial_data,
        "_perform_database_migration",
        perform_database_migration_mock,
    )
    perform_data_migrations_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        mlrun.api.initial_data, "_perform_data_migrations", perform_data_migrations_mock
    )

    for case in [
        # All 4 schema and data combinations with database and not from scratch
        {
            "database_migration": True,
            "schema_migration": False,
            "data_migration": False,
            "from_scratch": False,
        },
        {
            "database_migration": True,
            "schema_migration": True,
            "data_migration": False,
            "from_scratch": False,
        },
        {
            "database_migration": True,
            "schema_migration": True,
            "data_migration": True,
            "from_scratch": False,
        },
        {
            "database_migration": True,
            "schema_migration": False,
            "data_migration": True,
            "from_scratch": False,
        },
        # All 4 schema and data combinations with database and from scratch
        {
            "database_migration": True,
            "schema_migration": False,
            "data_migration": False,
            "from_scratch": True,
        },
        {
            "database_migration": True,
            "schema_migration": True,
            "data_migration": False,
            "from_scratch": True,
        },
        {
            "database_migration": True,
            "schema_migration": True,
            "data_migration": True,
            "from_scratch": True,
        },
        {
            "database_migration": True,
            "schema_migration": False,
            "data_migration": True,
            "from_scratch": True,
        },
        # No database, not from scratch, at least of schema and data 3 combinations
        {
            "database_migration": False,
            "schema_migration": True,
            "data_migration": False,
            "from_scratch": False,
        },
        {
            "database_migration": False,
            "schema_migration": False,
            "data_migration": True,
            "from_scratch": False,
        },
        {
            "database_migration": False,
            "schema_migration": True,
            "data_migration": True,
            "from_scratch": False,
        },
    ]:
        sqlite_migration_util_mock.return_value.is_database_migration_needed.return_value = case.get(
            "database_migration", False
        )
        alembic_util_mock.return_value.is_migration_from_scratch.return_value = (
            case.get("from_scratch", False)
        )
        alembic_util_mock.return_value.is_schema_migration_needed.return_value = (
            case.get("schema_migration", False)
        )
        is_latest_data_version_mock.return_value = not case.get("data_migration", False)

        mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.online
        mlrun.api.initial_data.init_data()
        failure_message = f"Failed in case: {case}"
        assert (
            mlrun.mlconf.httpdb.state
            == mlrun.api.schemas.APIStates.waiting_for_migrations
        ), failure_message
        # assert the api just changed state and no operation was done
        assert db_backup_util_mock.call_count == 0, failure_message
        assert perform_schema_migrations_mock.call_count == 0, failure_message
        assert perform_database_migration_mock.call_count == 0, failure_message
        assert perform_data_migrations_mock.call_count == 0, failure_message
