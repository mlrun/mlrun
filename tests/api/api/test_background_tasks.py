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
import datetime
import http
import typing
import unittest.mock

import fastapi
import fastapi.testclient
import pytest
import requests
import sqlalchemy.orm

import mlrun.api.api.deps
import mlrun.api.main
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.background_tasks
import mlrun.api.utils.clients.chief

test_router = fastapi.APIRouter()


# the reason we  have to declare an endpoint is that our class is built on top of FastAPI's background_tasks mechanism,
# and to get this class, we must trigger an endpoint
@test_router.post(
    "/projects/{project}/background-tasks",
    response_model=mlrun.api.schemas.BackgroundTask,
)
def create_project_background_task(
    project: str,
    background_tasks: fastapi.BackgroundTasks,
    failed_task: bool = False,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    function = bump_counter
    if failed_task:
        function = failing_function
    return mlrun.api.utils.background_tasks.ProjectBackgroundTasksHandler().create_background_task(
        db_session, project, background_tasks, function
    )


@test_router.post(
    "/internal-background-tasks",
    response_model=mlrun.api.schemas.BackgroundTask,
)
def create_internal_background_task(
    background_tasks: fastapi.BackgroundTasks,
    failed_task: bool = False,
):
    function = bump_counter
    if failed_task:
        function = failing_function
    return mlrun.api.utils.background_tasks.InternalBackgroundTasksHandler().create_background_task(
        background_tasks,
        function,
    )


call_counter: int = 0


async def bump_counter():
    global call_counter
    call_counter += 1


def failing_function():
    raise RuntimeError("I am a failure")


# must add it here since we're adding routes
@pytest.fixture()
def client() -> typing.Generator:
    mlrun.api.main.app.include_router(test_router, prefix="/test")
    with fastapi.testclient.TestClient(mlrun.api.main.app) as client:
        yield client


ORIGINAL_VERSIONED_API_PREFIX = mlrun.api.main.BASE_VERSIONED_API_PREFIX


def test_redirection_from_worker_to_chief_trigger_migrations(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, httpserver
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    task_name = "testy"

    for test_case in [
        {
            "expected_status": http.HTTPStatus.OK.value,
            "expected_body": {},
        },
        {
            "expected_status": http.HTTPStatus.ACCEPTED.value,
            "expected_body": _generate_background_task(task_name).json(),
        },
        {
            "expected_status": http.HTTPStatus.PRECONDITION_FAILED.value,
            "expected_body": {"detail": {"reason": "waiting for migrations"}},
        },
        {
            "expected_status": http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            "expected_body": {"detail": {"reason": "unexpected error"}},
        },
    ]:

        expected_status = test_case.get("expected_status")
        expected_response = test_case.get("expected_body")
        httpserver.expect_ordered_request(
            f"{ORIGINAL_VERSIONED_API_PREFIX}/operations/migrations", method="POST"
        ).respond_with_json(expected_response, status=expected_status)
        url = httpserver.url_for("")
        mlrun.mlconf.httpdb.clusterization.chief.url = url
        response = client.post(f"{ORIGINAL_VERSIONED_API_PREFIX}/operations/migrations")
        assert response.status_code == expected_status
        assert response.json() == expected_response


def test_redirection_from_worker_to_chief_get_internal_background_tasks(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, httpserver
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    task_name = "testy"
    for test_case in [
        {
            "expected_status": http.HTTPStatus.OK.value,
            "expected_body": _generate_background_task(task_name).json(),
        },
        {
            "expected_status": http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            "expected_body": {"detail": {"reason": "error_message"}},
        },
    ]:

        expected_status = test_case.get("expected_status")
        expected_response = test_case.get("expected_body")
        httpserver.expect_ordered_request(
            f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{task_name}",
            method="GET",
        ).respond_with_json(expected_response, status=expected_status)
        url = httpserver.url_for("")
        mlrun.mlconf.httpdb.clusterization.chief.url = url
        response = client.get(
            f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{task_name}"
        )
        assert response.status_code == expected_status
        assert response.json() == expected_response


def test_create_project_background_task_in_chief_success(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    project = "project"
    assert call_counter == 0
    response = client.post(f"/test/projects/{project}/background-tasks")
    background_task = _assert_background_task_creation(project, response)
    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state == mlrun.api.schemas.BackgroundTaskState.succeeded
    )
    assert background_task.metadata.updated is not None
    assert call_counter == 1


def test_create_project_background_task_in_chief_failure(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    project = "project"
    response = client.post(
        f"/test/projects/{project}/background-tasks", params={"failed_task": True}
    )
    background_task = _assert_background_task_creation(project, response)
    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.failed
    assert background_task.metadata.updated is not None


def test_get_project_background_task_not_exists(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    project = "project"
    name = "task-name"
    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}/background-tasks/{name}"
    )
    assert response.status_code == http.HTTPStatus.NOT_FOUND.value


def test_get_background_task_auth_skip(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    mlrun.api.utils.auth.verifier.AuthVerifier().query_resource_permissions = (
        unittest.mock.AsyncMock()
    )
    mlrun.mlconf.igz_version = "3.2.0-b26.20210904121245"
    response = client.post("/test/internal-background-tasks")
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    assert (
        mlrun.api.utils.auth.verifier.AuthVerifier().query_resource_permissions.call_count
        == 0
    )

    mlrun.mlconf.igz_version = "3.7.0-b26.20210904121245"
    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    assert (
        mlrun.api.utils.auth.verifier.AuthVerifier().query_resource_permissions.call_count
        == 1
    )


def test_get_internal_background_task_redirect_from_worker_to_chief_exists(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, monkeypatch
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    name = "task-name"
    expected_background_task = _generate_background_task(name)
    handler_mock = mlrun.api.utils.clients.chief.Client()
    handler_mock.get_internal_background_task = unittest.mock.Mock(
        return_value=expected_background_task
    )
    monkeypatch.setattr(
        mlrun.api.utils.clients.chief,
        "Client",
        lambda *args, **kwargs: handler_mock,
    )
    response = client.get(f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{name}")
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task == expected_background_task


def test_get_internal_background_task_from_worker_redirect_to_chief_doesnt_exists(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, monkeypatch
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    name = "task-name"
    handler_mock = mlrun.api.utils.clients.chief.Client()
    handler_mock.get_internal_background_task = unittest.mock.Mock(
        side_effect=mlrun.errors.MLRunHTTPError()
    )
    monkeypatch.setattr(
        mlrun.api.utils.clients.chief,
        "Client",
        lambda *args, **kwargs: handler_mock,
    )
    with pytest.raises(mlrun.errors.MLRunHTTPError):
        client.get(f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{name}")


def test_get_internal_background_task_in_chief_exists(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    response = client.post("/test/internal-background-tasks")
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.metadata.project is None

    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value


def test_trigger_migrations_from_worker_returns_same_response_as_chief(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, monkeypatch
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"

    for test_case in [
        {
            "status_code": http.HTTPStatus.PRECONDITION_FAILED.value,
            "content": b'{"detail":{"reason":"MLRunPreconditionFailedError(\'Migrations were'
            b" already triggered and failed. Restart the API to retry')\"}}",
        },
        {
            "status_code": http.HTTPStatus.ACCEPTED.value,
            "content": b'{"kind":"BackgroundTask","metadata":{"name":"2efd3890-3a12-416d-ae92-807b7796e257",'
            b'"project":null,"created":"2022-06-13T21:30:42.431158","updated":'
            b'"2022-06-13T21:30:42.431158","timeout":null},"spec":{},"status":{"state":"running"}}',
        },
        {
            "status_code": http.HTTPStatus.OK.value,
            "content": b"{}",
        },
        {
            "status_code": http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            "content": None,
        },
    ]:
        expected_response = fastapi.Response(
            status_code=test_case.get("status_code"), content=test_case.get("content")
        )
        handler_mock = mlrun.api.utils.clients.chief.Client()
        handler_mock.trigger_migrations = unittest.mock.Mock(
            return_value=expected_response
        )
        monkeypatch.setattr(
            mlrun.api.utils.clients.chief,
            "Client",
            lambda *args, **kwargs: handler_mock,
        )
        response: requests.Response = client.post(
            f"{ORIGINAL_VERSIONED_API_PREFIX}/operations/migrations"
        )
        assert response.status_code == expected_response.status_code
        assert response.content == expected_response.body


def _generate_background_task(
    background_task_name,
    state: mlrun.api.schemas.BackgroundTaskState = mlrun.api.schemas.BackgroundTaskState.running,
) -> mlrun.api.schemas.BackgroundTask:
    now = datetime.datetime.utcnow()
    return mlrun.api.schemas.BackgroundTask(
        metadata=mlrun.api.schemas.BackgroundTaskMetadata(
            name=background_task_name,
            created=now,
            updated=now,
        ),
        status=mlrun.api.schemas.BackgroundTaskStatus(state=state.value),
        spec=mlrun.api.schemas.BackgroundTaskSpec(),
    )


def _assert_background_task_creation(expected_project, response):
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.kind == mlrun.api.schemas.ObjectKind.background_task
    assert background_task.metadata.project == expected_project
    assert background_task.metadata.created is not None
    assert background_task.metadata.updated is not None
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    return background_task
