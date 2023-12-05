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
import http
import itertools
import unittest.mock

import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import server.api.initial_data
import server.api.utils.auth.verifier
import server.api.utils.db.alembic
import server.api.utils.db.backup


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
    assert (
        expected_message in response.text
    ), f"Expected message: {expected_message}, actual: {response.text}"


def test_init_data_migration_required_recognition(monkeypatch) -> None:
    alembic_util_mock = unittest.mock.Mock()
    monkeypatch.setattr(server.api.utils.db.alembic, "AlembicUtil", alembic_util_mock)
    is_latest_data_version_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        server.api.initial_data, "_is_latest_data_version", is_latest_data_version_mock
    )
    db_backup_util_mock = unittest.mock.Mock()
    monkeypatch.setattr(server.api.utils.db.backup, "DBBackupUtil", db_backup_util_mock)
    perform_schema_migrations_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        server.api.initial_data,
        "_perform_schema_migrations",
        perform_schema_migrations_mock,
    )
    perform_data_migrations_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        server.api.initial_data,
        "_perform_data_migrations",
        perform_data_migrations_mock,
    )
    for product in itertools.product([True, False], repeat=3):
        case = {
            "schema_migration": product[0],
            "data_migration": product[1],
            "from_scratch": product[2],
        }

        alembic_util_mock.return_value.is_migration_from_scratch.return_value = (
            case.get("from_scratch", False)
        )
        alembic_util_mock.return_value.is_schema_migration_needed.return_value = (
            case.get("schema_migration", False)
        )
        is_latest_data_version_mock.return_value = not case.get("data_migration", False)

        mlrun.mlconf.httpdb.state = mlrun.common.schemas.APIStates.online
        server.api.initial_data.init_data()
        failure_message = f"Failed in case: {case}"
        assert (
            mlrun.mlconf.httpdb.state
            == mlrun.common.schemas.APIStates.waiting_for_migrations
        ), failure_message
        # assert the api just changed state and no operation was done
        assert db_backup_util_mock.call_count == 0, failure_message
        assert perform_schema_migrations_mock.call_count == 0, failure_message
        assert perform_data_migrations_mock.call_count == 0, failure_message
