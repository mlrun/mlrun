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
import json

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


def test_get_background_task_from_chief_success(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    requests_mock: requests_mock_package.Mocker,
):
    task_name = "test-for-chief"
    background_schema = _generate_background_task(task_name)
    # using jsonable_encoder because datetime isn't json serializable object
    # https://fastapi.tiangolo.com/tutorial/encoder/
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    requests_mock.get(
        f"{api_url}/api/v1/background-tasks/{task_name}", json=response_body
    )
    response = chief_client.get_internal_background_task(task_name)
    assert response.status_code == http.HTTPStatus.OK
    background_task = _transform_response_to_background_task(response)
    assert background_task.metadata.name == task_name
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    assert background_task.metadata.created == background_schema.metadata.created

    background_schema.status.state = mlrun.api.schemas.BackgroundTaskState.succeeded
    background_schema.metadata.updated = datetime.datetime.utcnow()
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    requests_mock.get(
        f"{api_url}/api/v1/background-tasks/{task_name}", json=response_body
    )
    response = chief_client.get_internal_background_task(task_name)
    assert response.status_code == http.HTTPStatus.OK
    background_task = _transform_response_to_background_task(response)
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
    requests_mock.get(
        f"{api_url}/api/v1/background-tasks/{task_name}",
        status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
    )
    response = chief_client.get_internal_background_task(task_name)
    assert response.status_code == http.HTTPStatus.INTERNAL_SERVER_ERROR.value


def test_trigger_migration_succeeded(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    requests_mock: requests_mock_package.Mocker,
):
    task_name = "test-for-chief"
    background_schema = _generate_background_task(task_name)
    # using jsonable_encoder because datetime isn't json serializable object
    # https://fastapi.tiangolo.com/tutorial/encoder/
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    requests_mock.post(
        f"{api_url}/api/v1/operations/migrations",
        json=response_body,
        status_code=http.HTTPStatus.ACCEPTED,
    )
    response = chief_client.trigger_migrations()
    assert response.status_code == http.HTTPStatus.ACCEPTED
    background_task = _transform_response_to_background_task(response)
    assert background_task.metadata.name == task_name
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    assert background_task.metadata.created == background_schema.metadata.created

    background_schema.status.state = mlrun.api.schemas.BackgroundTaskState.succeeded
    background_schema.metadata.updated = datetime.datetime.utcnow()
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    requests_mock.post(
        f"{api_url}/api/v1/operations/migrations",
        json=response_body,
        status_code=http.HTTPStatus.ACCEPTED,
    )
    response = chief_client.trigger_migrations()
    assert response.status_code == http.HTTPStatus.ACCEPTED
    background_task = _transform_response_to_background_task(response)
    assert background_task.metadata.name == task_name
    assert (
        background_task.status.state == mlrun.api.schemas.BackgroundTaskState.succeeded
    )
    assert background_task.metadata.created == background_schema.metadata.created
    assert background_task.metadata.updated == background_schema.metadata.updated
    assert background_task.metadata.updated > background_task.metadata.created


def test_trigger_migrations_from_chief_failures(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    requests_mock: requests_mock_package.Mocker,
):
    requests_mock.post(
        f"{api_url}/api/v1/operations/migrations",
        status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
    )
    response = chief_client.trigger_migrations()
    assert response.status_code == http.HTTPStatus.INTERNAL_SERVER_ERROR.value
    assert not response.body

    requests_mock.post(
        f"{api_url}/api/v1/operations/migrations",
        status_code=http.HTTPStatus.PRECONDITION_FAILED.value,
        text="Migrations were already triggered and failed. Restart the API to retry",
    )
    response = chief_client.trigger_migrations()
    assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
    assert "Migrations were already triggered and failed" in response.body.decode(
        "utf-8"
    )


def test_trigger_migrations_chief_restarted_while_executing_migrations(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    requests_mock: requests_mock_package.Mocker,
):
    task_name = "test-bg-failed"

    background_schema = _generate_background_task(task_name)
    # using jsonable_encoder because datetime isn't json serializable object
    # https://fastapi.tiangolo.com/tutorial/encoder/
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    requests_mock.post(
        f"{api_url}/api/v1/operations/migrations",
        json=response_body,
        status_code=http.HTTPStatus.ACCEPTED,
    )
    response = chief_client.trigger_migrations()
    assert response.status_code == http.HTTPStatus.ACCEPTED
    background_task = _transform_response_to_background_task(response)
    assert background_task.metadata.name == task_name
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    assert background_task.metadata.created == background_schema.metadata.created

    # in internal background tasks, failed state is only when the background task doesn't exists in memory,
    # which means the api was restarted
    background_schema.status.state = mlrun.api.schemas.BackgroundTaskState.failed
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    requests_mock.get(
        f"{api_url}/api/v1/background-tasks/{task_name}", json=response_body
    )
    response = chief_client.get_internal_background_task(task_name)
    assert response.status_code == http.HTTPStatus.OK
    background_task = _transform_response_to_background_task(response)
    assert background_task.metadata.name == task_name
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.failed
    assert background_task.metadata.created == background_schema.metadata.created


def _transform_response_to_background_task(response: fastapi.Response):
    decoded_body = response.body.decode("utf-8")
    body_dict = json.loads(decoded_body)
    return mlrun.api.schemas.BackgroundTask(**body_dict)


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
