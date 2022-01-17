import http
import unittest.mock

import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.initial_data
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier


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
        assert response.status_code == http.HTTPStatus.OK.value

        response = client.get("projects")
        assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
        assert expected_message in response.text


def test_init_data_migration_required_recognition() -> None:
    # mock that migration is needed
    original_is_migration_needed = mlrun.api.initial_data._is_migration_needed
    mlrun.api.initial_data._is_migration_needed = unittest.mock.Mock(return_value=True)
    mlrun.api.initial_data.init_data()
    assert (
        mlrun.mlconf.httpdb.state == mlrun.api.schemas.APIStates.waiting_for_migrations
    )
    mlrun.api.initial_data._is_migration_needed = original_is_migration_needed
