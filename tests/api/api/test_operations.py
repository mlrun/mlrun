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
from datetime import datetime

import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun
import mlrun.api.api.endpoints.operations
import mlrun.api.crud
import mlrun.api.initial_data
import mlrun.api.schemas
import mlrun.api.utils.background_tasks
import mlrun.api.utils.clients.iguazio
import mlrun.api.utils.singletons.scheduler
import mlrun.errors
import mlrun.runtimes
from mlrun.utils import logger


def test_migrations_already_in_progress(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, monkeypatch
) -> None:
    background_task_name = "some-name"
    mlrun.api.api.endpoints.operations.current_migration_background_task_name = (
        background_task_name
    )
    handler_mock = mlrun.api.utils.background_tasks.InternalBackgroundTasksHandler()
    handler_mock.get_background_task = unittest.mock.Mock(
        return_value=(_generate_background_task_schema(background_task_name))
    )
    monkeypatch.setattr(
        mlrun.api.utils.background_tasks,
        "InternalBackgroundTasksHandler",
        lambda *args, **kwargs: handler_mock,
    )
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.migrations_in_progress
    response = client.post("operations/migrations")
    assert response.status_code == http.HTTPStatus.ACCEPTED.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task_name == background_task.metadata.name
    mlrun.api.api.endpoints.operations.current_migration_background_task_name = None


def test_migrations_failed(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.migrations_failed
    response = client.post("operations/migrations")
    assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
    assert "Migrations were already triggered and failed" in response.text


def test_migrations_not_needed(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.online
    response = client.post("operations/migrations")
    assert response.status_code == http.HTTPStatus.OK.value


def _mock_migration_process(*args, **kwargs):
    logger.info("Mocking migration process")
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.migrations_completed


@pytest.fixture
def _mock_waiting_for_migration():
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.waiting_for_migrations


def test_migrations_success(
    # db calls init_data with from_scratch=True which means it will anyways do the migrations
    # therefore in order to make the api to be started as if its in a state where migrations are needed
    # we just add a middle fixture that sets the state
    db: sqlalchemy.orm.Session,
    _mock_waiting_for_migration,
    client: fastapi.testclient.TestClient,
) -> None:
    original_init_data = mlrun.api.initial_data.init_data
    mlrun.api.initial_data.init_data = _mock_migration_process
    response = client.get("projects")
    # error cause we're waiting for migrations
    assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
    assert "API is waiting for migrations to be triggered" in response.text
    # not initialized until we're not doing migrations
    assert mlrun.api.utils.singletons.scheduler.get_scheduler() is None
    # trigger migrations
    response = client.post("operations/migrations")
    assert response.status_code == http.HTTPStatus.ACCEPTED.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    response = client.get(f"background-tasks/{background_task.metadata.name}")
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state == mlrun.api.schemas.BackgroundTaskState.succeeded
    )
    assert mlrun.mlconf.httpdb.state == mlrun.api.schemas.APIStates.online
    # now we should be able to get projects
    response = client.get("projects")
    assert response.status_code == http.HTTPStatus.OK.value
    # should be initialized
    assert mlrun.api.utils.singletons.scheduler.get_scheduler() is not None

    # tear down
    mlrun.api.initial_data.init_data = original_init_data


def _generate_background_task_schema(
    background_task_name,
) -> mlrun.api.schemas.BackgroundTask:
    return mlrun.api.schemas.BackgroundTask(
        metadata=mlrun.api.schemas.BackgroundTaskMetadata(
            name=background_task_name,
            created=datetime.utcnow(),
            updated=datetime.utcnow(),
        ),
        status=mlrun.api.schemas.BackgroundTaskStatus(
            state=mlrun.api.schemas.BackgroundTaskState.running
        ),
        spec=mlrun.api.schemas.BackgroundTaskSpec(),
    )
