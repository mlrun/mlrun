import datetime
import http
import typing
import unittest.mock

import fastapi
import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.api.api.deps
import mlrun.api.main
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.background_tasks
import mlrun.api.utils.clients.chief

test_router = fastapi.APIRouter()


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
    return mlrun.api.utils.background_tasks.Handler().create_background_task(
        db_session, background_tasks, function, project
    )


@test_router.post(
    "/background-tasks",
    response_model=mlrun.api.schemas.BackgroundTask,
)
def create_background_task(
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
        db_session, background_tasks, function
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


@pytest.mark.parametrize("role", ["worker, chief"])
def test_create_background_task(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, role
):
    mlrun.mlconf.httpdb.clusterization.role = role
    response = client.post("/test/background-tasks")
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.metadata.project is None

    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.metadata.project is None
    assert background_task.metadata.timeout is None


def test_get_background_task_auth_skip(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    mlrun.api.utils.auth.verifier.AuthVerifier().query_resource_permissions = (
        unittest.mock.Mock()
    )
    mlrun.mlconf.igz_version = "3.2.0-b26.20210904121245"
    response = client.post("/test/background-tasks")
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

    mlrun.mlconf.igz_version = "3.5.0-b26.20210904121245"
    response = client.get(
        f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    assert (
        mlrun.api.utils.auth.verifier.AuthVerifier().query_resource_permissions.call_count
        == 1
    )


def test_get_background_task_not_exists_on_worker_exists_in_chief(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, monkeypatch
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    name = "task-name"
    expected_background_task = _generate_background_task_schema(name)
    handler_mock = mlrun.api.utils.clients.chief.Client()
    handler_mock.get_background_task = unittest.mock.Mock(
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


def test_get_background_task_not_exists_in_both_worker_and_chief(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, monkeypatch
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    name = "task-name"
    handler_mock = mlrun.api.utils.clients.chief.Client()
    handler_mock.get_background_task = unittest.mock.Mock(
        side_effect=mlrun.errors.MLRunHTTPError("Explode")
    )
    monkeypatch.setattr(
        mlrun.api.utils.clients.chief,
        "Client",
        lambda *args, **kwargs: handler_mock,
    )
    with pytest.raises(mlrun.errors.MLRunHTTPError):
        client.get(f"{ORIGINAL_VERSIONED_API_PREFIX}/background-tasks/{name}")


def test_get_background_task_in_chief_exists_in_memory(
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


def _generate_background_task_schema(
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
