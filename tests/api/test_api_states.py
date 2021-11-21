import http

import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import pytest


def test_offline_state(
        db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.offline
    response = client.get(
        "/api/healthz"
    )
    assert response.status_code == http.HTTPStatus.OK.value

    response = client.get(
        "/api/projects"
    )
    assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
    assert "API is in offline state" in response.text


def test_waiting_for_migrations_state(
        db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.waiting_for_migrations
    response = client.get(
        "/api/healthz"
    )
    assert response.status_code == http.HTTPStatus.OK.value

    response = client.get(
        "/api/projects/some-project/background-tasks/some-task"
    )
    assert response.status_code == http.HTTPStatus.OK.value

    response = client.get(
        "/api/projects"
    )
    assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
    assert "API is waiting for migration to be triggered" in response.text
