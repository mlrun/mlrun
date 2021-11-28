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
    response = client.get("/api/healthz")
    assert response.status_code == http.HTTPStatus.OK.value

    response = client.get("/api/projects")
    assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
    assert "API is in offline state" in response.text


def test_waiting_for_migrations_state(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.waiting_for_migrations
    response = client.get("/api/healthz")
    assert response.status_code == http.HTTPStatus.OK.value

    response = client.get("/api/projects/some-project/background-tasks/some-task")
    assert response.status_code == http.HTTPStatus.OK.value

    response = client.get("/api/projects")
    assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
    assert "API is waiting for migration to be triggered" in response.text


def test_init_data_migration_required_recognition() -> None:
    # mock that migration is needed
    original_is_migration_needed = mlrun.api.initial_data._is_migration_needed
    mlrun.api.initial_data._is_migration_needed = unittest.mock.Mock(return_value=True)
    mlrun.api.initial_data.init_data()
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.waiting_for_migrations
    mlrun.api.initial_data._is_migration_needed = original_is_migration_needed
