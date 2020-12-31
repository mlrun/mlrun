import http
import typing

import fastapi
import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.api.api.deps
import mlrun.api.main
import mlrun.api.schemas
import mlrun.api.utils.background_tasks

test_router = fastapi.APIRouter()


@test_router.post(
    "/projects/{project}/background-tasks",
    response_model=mlrun.api.schemas.BackgroundTask,
)
def create_background_task(
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
        db_session, project, background_tasks, function
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


def test_create_background_task_success(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    project = "project"
    assert call_counter == 0
    response = client.post(f"/test/projects/{project}/background-tasks")
    background_task = _assert_background_task_creation(project, response)
    response = client.get(
        f"/api/projects/{project}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state == mlrun.api.schemas.BackgroundTaskState.succeeded
    )
    assert background_task.metadata.updated is not None
    assert call_counter == 1


def test_create_background_task_failure(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    project = "project"
    response = client.post(
        f"/test/projects/{project}/background-tasks", params={"failed_task": True}
    )
    background_task = _assert_background_task_creation(project, response)
    response = client.get(
        f"/api/projects/{project}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.failed
    assert background_task.metadata.updated is not None


def test_get_background_task_not_exists(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    project = "project"
    name = "task-name"
    response = client.get(f"/api/projects/{project}/background-tasks/{name}")
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.metadata.project == project
    assert background_task.metadata.name == name
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.failed


def _assert_background_task_creation(expected_project, response):
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.kind == mlrun.api.schemas.ObjectKind.background_task
    assert background_task.metadata.project == expected_project
    assert background_task.metadata.created is not None
    assert background_task.metadata.updated is None
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    return background_task
