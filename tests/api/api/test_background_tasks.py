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

import asyncio
import datetime
import http
import typing
import unittest.mock
from concurrent.futures import ThreadPoolExecutor

import fastapi
import fastapi.concurrency
import fastapi.testclient
import httpx
import pytest
import pytest_asyncio
import requests
import sqlalchemy.orm

import mlrun.common.schemas
import server.api.api.deps
import server.api.utils.auth.verifier
import server.api.utils.background_tasks
import server.api.utils.clients.chief
from mlrun.utils import logger
from server.api import main

test_router = fastapi.APIRouter()


# the reason we  have to declare an endpoint is that our class is built on top of FastAPI's background_tasks mechanism,
# and to get this class, we must trigger an endpoint
@test_router.post(
    "/projects/{project}/background-tasks",
    response_model=mlrun.common.schemas.BackgroundTask,
)
async def create_project_background_task(
    project: str,
    background_tasks: fastapi.BackgroundTasks,
    failed_task: bool = False,
    timeout: int = None,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
):
    function = bump_counter
    args = []
    if failed_task:
        function = failing_function
    # add timeout to test failing the background task due to timeout
    elif timeout:
        function = long_function
        # adds some time to make sure that it sleeps longer than the timeout
        args = [timeout + 3]
    return await fastapi.concurrency.run_in_threadpool(
        server.api.utils.background_tasks.ProjectBackgroundTasksHandler().create_background_task,
        db_session,
        project,
        background_tasks,
        function,
        timeout,
        None,
        *args,
    )


@test_router.post(
    "/internal-background-tasks",
    response_model=mlrun.common.schemas.BackgroundTask,
)
def create_internal_background_task(
    background_tasks: fastapi.BackgroundTasks,
    failed_task: bool = False,
    project: str = None,
):
    function = bump_counter
    if failed_task:
        function = failing_function

    if project:
        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name=project)
        )
    (
        task,
        task_name,
    ) = server.api.utils.background_tasks.InternalBackgroundTasksHandler().create_background_task(
        "bump_counter", None, function, project=project
    )
    background_tasks.add_task(task)
    return server.api.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task(
        task_name
    )


@test_router.post(
    "/long-internal-background-tasks/{timeout}",
    response_model=mlrun.common.schemas.BackgroundTask,
)
def create_long_internal_background_task(
    background_tasks: fastapi.BackgroundTasks,
    timeout: int = 5,
):
    (
        task,
        task_name,
    ) = server.api.utils.background_tasks.InternalBackgroundTasksHandler().create_background_task(
        "long_bump_counter", None, long_function, sleep_time=timeout
    )
    background_tasks.add_task(task)
    return server.api.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task(
        task_name
    )


call_counter: int = 0


async def bump_counter():
    global call_counter
    call_counter += 1


def failing_function():
    raise RuntimeError("I am a failure")


async def long_function(sleep_time):
    await asyncio.sleep(sleep_time)
    await bump_counter()


# must add it here since we're adding routes
@pytest.fixture()
def client() -> typing.Generator:
    main.app.include_router(test_router, prefix="/test")
    with fastapi.testclient.TestClient(main.app) as client:
        yield client


class ThreadedAsyncClient(httpx.AsyncClient):
    async def post(self, *args, **kwargs):
        thread_pool_executor = ThreadPoolExecutor(1)
        async_event_loop = asyncio.new_event_loop()
        thread_pool_executor.submit(asyncio.set_event_loop, async_event_loop).result()
        future = thread_pool_executor.submit(
            async_event_loop.run_until_complete, super().post(*args, **kwargs)
        )
        # release the current event loop to let the other thread kick in
        await asyncio.sleep(1)
        return future


@pytest_asyncio.fixture
async def async_client() -> typing.Iterator[ThreadedAsyncClient]:
    """
    Async client that runs in a separate thread.
    When posting with the client, the request is sent on a different thread, and the method returns a future.
    To get the response, call result() on the future.
    Example:
        result = await async_client.post(...)
        response = result.result()
    """
    main.app.include_router(test_router, prefix="/test")
    async with ThreadedAsyncClient(app=main.app, base_url="https://mlrun") as client:
        yield client


ORIGINAL_VERSIONED_API_PREFIX = main.BASE_VERSIONED_API_PREFIX


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
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state
        == mlrun.common.schemas.BackgroundTaskState.succeeded
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
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state == mlrun.common.schemas.BackgroundTaskState.failed
    )
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


def test_get_internal_background_task_auth(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    server.api.utils.auth.verifier.AuthVerifier().query_project_permissions = (
        unittest.mock.AsyncMock()
    )
    response = client.post("/test/internal-background-tasks?project=my-proj")
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    assert (
        server.api.utils.auth.verifier.AuthVerifier().query_project_permissions.call_count
        == 0
    )

    # Create another task without a project should skip authz
    response = client.post("/test/internal-background-tasks")
    assert response.status_code == http.HTTPStatus.OK.value

    response = client.get(f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks")
    assert response.status_code == http.HTTPStatus.OK.value
    assert (
        server.api.utils.auth.verifier.AuthVerifier().query_project_permissions.call_count
        == 0
    )


def test_get_internal_background_task_redirect_from_worker_to_chief_exists(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, monkeypatch
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    name = "task-name"
    expected_background_task = _generate_background_task(name)
    handler_mock = server.api.utils.clients.chief.Client()
    handler_mock.get_internal_background_task = unittest.mock.AsyncMock(
        return_value=expected_background_task
    )
    monkeypatch.setattr(
        server.api.utils.clients.chief,
        "Client",
        lambda *args, **kwargs: handler_mock,
    )
    response = client.get(f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{name}")
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    assert background_task == expected_background_task


def test_get_internal_background_task_from_worker_redirect_to_chief_doesnt_exists(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, monkeypatch
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    name = "task-name"
    handler_mock = server.api.utils.clients.chief.Client()
    handler_mock.get_internal_background_task = unittest.mock.AsyncMock(
        side_effect=mlrun.errors.MLRunHTTPError()
    )
    monkeypatch.setattr(
        server.api.utils.clients.chief,
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
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    assert background_task.metadata.project is None

    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value


@pytest.mark.asyncio
async def test_internal_background_task_already_running(
    db: sqlalchemy.orm.Session, async_client: httpx.AsyncClient
):
    timeout = 3
    curr_call_counter = call_counter

    # if we await the first future before sending the second request, the second request will be sent after the whole
    # background task is finished because of how httpx.AsyncClient works. To avoid this, we send both requests and
    # await them together. This way, the second request will be sent before the first background task is finished and
    # will be rejected.
    first_future = await async_client.post(
        f"/test/long-internal-background-tasks/{timeout}"
    )
    second_future = await async_client.post(
        f"/test/long-internal-background-tasks/{timeout}"
    )
    first_response = first_future.result()
    second_response = second_future.result()
    assert first_response.status_code == http.HTTPStatus.OK.value
    assert second_response.status_code == http.HTTPStatus.CONFLICT.value

    while curr_call_counter == call_counter:
        logger.info("Waiting for background task to finish")
        await asyncio.sleep(1)

    third_future = await async_client.post(
        f"/test/long-internal-background-tasks/{timeout}"
    )
    third_response = third_future.result()
    assert third_response.status_code == http.HTTPStatus.OK.value
    while curr_call_counter == call_counter + 1:
        logger.info("Waiting for background task to finish")
        await asyncio.sleep(1)


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
        handler_mock = server.api.utils.clients.chief.Client()
        handler_mock.trigger_migrations = unittest.mock.AsyncMock(
            return_value=expected_response
        )
        monkeypatch.setattr(
            server.api.utils.clients.chief,
            "Client",
            lambda *args, **kwargs: handler_mock,
        )
        response: requests.Response = client.post(
            f"{ORIGINAL_VERSIONED_API_PREFIX}/operations/migrations"
        )
        assert response.status_code == expected_response.status_code
        assert response.content == expected_response.body


def test_list_project_background_tasks(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    project = "project"
    curr_call_counter = call_counter

    # list no background tasks
    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}/background-tasks"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_tasks = mlrun.common.schemas.BackgroundTaskList(**response.json())
    assert len(background_tasks.background_tasks) == 0

    # create 3 background tasks
    for i in range(3):
        response = client.post(f"/test/projects/{project}/background-tasks")
        _assert_background_task_creation(project, response)

    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}/background-tasks"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_tasks = mlrun.common.schemas.BackgroundTaskList(**response.json())
    assert len(background_tasks.background_tasks) == 3

    for background_task in background_tasks.background_tasks:
        assert (
            background_task.status.state
            == mlrun.common.schemas.BackgroundTaskState.succeeded
        )
        assert background_task.metadata.updated is not None

    assert call_counter == curr_call_counter + 3


@pytest.mark.asyncio
async def test_list_timed_out_project_background_task(
    db: sqlalchemy.orm.Session, async_client: httpx.AsyncClient
):
    project = "my-project"
    # create a background task that will not time out
    await async_client.post(f"/test/projects/{project}/background-tasks?timeout=5")

    # create a background task that will time out
    await async_client.post(f"/test/projects/{project}/background-tasks?timeout=1")

    # sleep pass the short timeout
    await asyncio.sleep(1)
    response = await async_client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}/background-tasks"
    )

    assert response.status_code == http.HTTPStatus.OK.value
    background_tasks = mlrun.common.schemas.BackgroundTaskList(**response.json())
    assert len(background_tasks.background_tasks) == 2

    for background_task in background_tasks.background_tasks:
        if background_task.metadata.timeout == 1:
            assert (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.failed
            )
        else:
            assert (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.running
            )


def _generate_background_task(
    background_task_name,
    state: mlrun.common.schemas.BackgroundTaskState = mlrun.common.schemas.BackgroundTaskState.running,
) -> mlrun.common.schemas.BackgroundTask:
    now = datetime.datetime.utcnow()
    return mlrun.common.schemas.BackgroundTask(
        metadata=mlrun.common.schemas.BackgroundTaskMetadata(
            name=background_task_name,
            created=now,
            updated=now,
        ),
        status=mlrun.common.schemas.BackgroundTaskStatus(state=state.value),
        spec=mlrun.common.schemas.BackgroundTaskSpec(),
    )


def _assert_background_task_creation(expected_project, response):
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    assert background_task.kind == mlrun.common.schemas.ObjectKind.background_task
    assert background_task.metadata.project == expected_project
    assert background_task.metadata.created is not None
    assert background_task.metadata.updated is not None
    assert (
        background_task.status.state == mlrun.common.schemas.BackgroundTaskState.running
    )
    return background_task
