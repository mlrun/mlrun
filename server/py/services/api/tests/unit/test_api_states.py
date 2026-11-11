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

import http
import unittest.mock

import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.common.db.dialects
import mlrun.common.schemas

import framework.db.sqldb
import services.api.initial_data
import services.api.utils.db.alembic
import services.api.utils.db.backup


def test_offline_state(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.mlconf.httpdb.state = mlrun.common.schemas.APIStates.offline
    response = client.get("healthz")
    assert response.status_code == http.HTTPStatus.SERVICE_UNAVAILABLE.value

    response = client.get("projects")
    assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
    assert "API is in offline state" in response.text


@pytest.mark.parametrize(
    "state, expected_healthz_status_code",
    [
        (
            mlrun.common.schemas.APIStates.waiting_for_migrations,
            http.HTTPStatus.OK.value,
        ),
        (
            mlrun.common.schemas.APIStates.migrations_in_progress,
            http.HTTPStatus.OK.value,
        ),
        (mlrun.common.schemas.APIStates.migrations_failed, http.HTTPStatus.OK.value),
        (
            mlrun.common.schemas.APIStates.waiting_for_chief,
            http.HTTPStatus.SERVICE_UNAVAILABLE.value,
        ),
    ],
)
def test_api_states(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    state,
    expected_healthz_status_code,
) -> None:
    mlrun.mlconf.httpdb.state = state
    response = client.get("healthz")
    assert response.status_code == expected_healthz_status_code

    response = client.get("projects/some-project/background-tasks/some-task")
    assert response.status_code == http.HTTPStatus.NOT_FOUND.value

    response = client.get("client-spec")
    assert response.status_code == http.HTTPStatus.OK.value

    response = client.get("projects")
    expected_message = mlrun.common.schemas.APIStates.description(state)
    assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
    assert expected_message in response.text, (
        f"Expected message: {expected_message}, actual: {response.text}"
    )


@pytest.mark.parametrize("schema_migration", [True, False])
@pytest.mark.parametrize("data_migration", [True, False])
def test_init_data_migration_required_recognition(
    db: sqlalchemy.orm.Session,
    monkeypatch,
    schema_migration,
    data_migration,
) -> None:
    # Simulate a MySQL engine with existing tables
    dummy_engine = unittest.mock.Mock()
    dummy_engine.dialect = mlrun.common.db.dialects.Dialects.MYSQL
    dummy_engine.url = "mysql://test"
    monkeypatch.setattr(
        framework.db.sqldb.sql_session,
        "get_engine",
        lambda: dummy_engine,
    )

    # Database exists and has tables
    monkeypatch.setattr(
        services.api.initial_data.sqlalchemy_utils,
        "database_exists",
        lambda url: True,
    )
    monkeypatch.setattr(
        services.api.initial_data.sqlalchemy,
        "inspect",
        lambda eng: type(
            "I",
            (),
            {"get_table_names": lambda self: ["fake_table"]},
        )(),
    )

    # Stub DBUtil to avoid real connectivity
    db_util_inst = unittest.mock.Mock()
    monkeypatch.setattr(
        services.api.initial_data,
        "DBUtil",
        lambda: db_util_inst,
    )

    # Stub AlembicUtil and data-version check
    alembic_inst = unittest.mock.Mock()
    monkeypatch.setattr(
        services.api.utils.db.alembic,
        "AlembicUtil",
        unittest.mock.Mock(return_value=alembic_inst),
    )
    alembic_inst.is_schema_migration_needed.return_value = schema_migration
    monkeypatch.setattr(
        services.api.initial_data,
        "_is_latest_data_version",
        lambda: not data_migration,
    )

    # Stub backup and migration routines
    monkeypatch.setattr(
        services.api.utils.db.backup,
        "DBBackupUtil",
        unittest.mock.Mock(),
    )
    perform_schema_migrations = unittest.mock.Mock()
    monkeypatch.setattr(
        services.api.initial_data,
        "_perform_schema_migrations",
        perform_schema_migrations,
    )
    perform_data_migrations = unittest.mock.Mock()
    monkeypatch.setattr(
        services.api.initial_data,
        "_perform_data_migrations",
        perform_data_migrations,
    )

    # Start from online state
    mlrun.mlconf.httpdb.state = mlrun.common.schemas.APIStates.online
    # Run init_data (default perform_migrations_if_needed=False)
    services.api.initial_data.init_data()

    # Expect waiting_for_migrations if any migration needed, otherwise remain online
    expected = (
        mlrun.common.schemas.APIStates.waiting_for_migrations
        if (schema_migration or data_migration)
        else mlrun.common.schemas.APIStates.online
    )
    assert mlrun.mlconf.httpdb.state == expected

    # No backup or migrations should have been executed in this recognition-only path
    assert services.api.utils.db.backup.DBBackupUtil.call_count == 0
    assert perform_schema_migrations.call_count == 0
    assert perform_data_migrations.call_count == 0
