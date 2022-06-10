import datetime

import fastapi.encoders
import pytest
import requests_mock as requests_mock_package

import mlrun.api.schemas
import mlrun.api.utils.clients.chief
import mlrun.config
import mlrun.errors


@pytest.fixture()
async def api_url() -> str:
    api_url = "http://chief-api.default-tenant.svc.cluster.local"
    mlrun.config.config.httpdb.clusterization.chief.url = api_url
    return api_url


@pytest.fixture()
async def chief_client(
    api_url: str,
) -> mlrun.api.utils.clients.chief.Client:
    client = mlrun.api.utils.clients.chief.Client()
    # force running init again so the configured api url will be used
    client.__init__()
    return client


def test_get_background_task_from_chief_succeeded(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    requests_mock: requests_mock_package.Mocker,
):
    task_name = "test-for-chief"
    background_schema = _generate_background_task_schema(task_name)
    # using jsonable_encoder because datetime isn't json serializable object
    # https://fastapi.tiangolo.com/tutorial/encoder/
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    requests_mock.get(
        f"{api_url}/api/v1/background-tasks/{task_name}", json=response_body
    )
    background_task = chief_client.get_background_task(task_name)
    assert background_task.metadata.name == task_name
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    assert background_task.metadata.created == background_schema.metadata.created

    background_schema.status.state = mlrun.api.schemas.BackgroundTaskState.succeeded
    background_schema.metadata.updated = datetime.datetime.utcnow()
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    requests_mock.get(
        f"{api_url}/api/v1/background-tasks/{task_name}", json=response_body
    )
    background_task = chief_client.get_background_task(task_name)
    assert background_task.metadata.name == task_name
    assert (
        background_task.status.state == mlrun.api.schemas.BackgroundTaskState.succeeded
    )
    assert background_task.metadata.created == background_schema.metadata.created
    assert background_task.metadata.updated == background_schema.metadata.updated
    assert background_task.metadata.updated > background_task.metadata.created


def test_get_background_task_from_chief_failed(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    requests_mock: requests_mock_package.Mocker,
):
    task_name = "test-for-chief"
    requests_mock.get(f"{api_url}/api/v1/background-tasks/{task_name}", status_code=500)
    with pytest.raises(Exception):
        chief_client.get_background_task(task_name)


def test_trigger_migration_succeeded(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    requests_mock: requests_mock_package.Mocker,
):
    task_name = "test-for-chief"
    background_schema = _generate_background_task_schema(task_name)
    # using jsonable_encoder because datetime isn't json serializable object
    # https://fastapi.tiangolo.com/tutorial/encoder/
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    requests_mock.post(f"{api_url}/api/v1/operations/migrations", json=response_body)
    background_task = chief_client.trigger_migrations()
    assert background_task.metadata.name == task_name
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    assert background_task.metadata.created == background_schema.metadata.created

    background_schema.status.state = mlrun.api.schemas.BackgroundTaskState.succeeded
    background_schema.metadata.updated = datetime.datetime.utcnow()
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    requests_mock.post(f"{api_url}/api/v1/operations/migrations", json=response_body)
    background_task = chief_client.trigger_migrations()
    assert background_task.metadata.name == task_name
    assert (
        background_task.status.state == mlrun.api.schemas.BackgroundTaskState.succeeded
    )
    assert background_task.metadata.created == background_schema.metadata.created
    assert background_task.metadata.updated == background_schema.metadata.updated
    assert background_task.metadata.updated > background_task.metadata.created


def test_trigger_migrations_from_chief_failed(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    requests_mock: requests_mock_package.Mocker,
):
    requests_mock.get(f"{api_url}/api/v1/operations/migrations", status_code=500)
    with pytest.raises(Exception):
        chief_client.trigger_migrations()


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
