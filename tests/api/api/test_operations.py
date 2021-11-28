import http

import fastapi.testclient
import sqlalchemy.orm

import mlrun
import mlrun.api.crud
import mlrun.api.api.endpoints.operations
import mlrun.api.initial_data
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.errors
import mlrun.runtimes
from mlrun.utils import logger


def test_start_migration_already_in_progress(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    background_task_name = "some-name"
    mlrun.api.api.endpoints.operations.current_migration_background_task_name = background_task_name
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.migrations_in_progress
    response = client.post("/api/operations/migrations")
    assert response.status_code == http.HTTPStatus.ACCEPTED.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task_name == background_task.metadata.name
    mlrun.api.api.endpoints.operations.current_migration_background_task_name = None


def test_start_migration_not_needed(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.online
    response = client.post("/api/operations/migrations")
    assert response.status_code == http.HTTPStatus.OK.value


def _mock_migration_process(*args, **kwargs):
    logger.info("Mocking migration process")
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.migrations_completed


def test_start_migration_success(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.waiting_for_migrations
    original_init_data = mlrun.api.initial_data.init_data
    mlrun.api.initial_data.init_data = _mock_migration_process
    response = client.post("/api/operations/migrations")
    assert response.status_code == http.HTTPStatus.ACCEPTED.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    response = client.get(f"/api/background-tasks/{background_task.metadata.name}")
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state == mlrun.api.schemas.BackgroundTaskState.succeeded
    )
    assert mlrun.mlconf.httpdb.state == mlrun.api.schemas.APIStates.online
    mlrun.api.initial_data.init_data = original_init_data
