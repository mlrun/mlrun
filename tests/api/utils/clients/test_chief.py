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
import unittest.mock

import aiohttp
import aiohttp.web
import fastapi.encoders
import pytest
import starlette.datastructures
from aiohttp import ClientConnectorError
from aiohttp.test_utils import TestClient, TestServer

import mlrun.api.schemas
import mlrun.api.utils.clients.chief
import mlrun.config
import mlrun.errors
from tests.common_fixtures import aioresponses_mock


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

    try:
        yield client
    finally:
        if client._session:
            await client._session.close()


@pytest.mark.asyncio
async def test_get_background_task_from_chief_success(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    aioresponses_mock: aioresponses_mock,
):
    task_name = "test-for-chief"
    background_schema = _generate_background_task(task_name)
    # using jsonable_encoder because datetime isn't json serializable object
    # https://fastapi.tiangolo.com/tutorial/encoder/
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    aioresponses_mock.get(
        f"{api_url}/api/v1/background-tasks/{task_name}", payload=response_body
    )
    response = await chief_client.get_internal_background_task(task_name)
    assert response.status_code == http.HTTPStatus.OK
    background_task = _transform_response_to_background_task(response)
    assert background_task.metadata.name == task_name
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    assert background_task.metadata.created == background_schema.metadata.created

    background_schema.status.state = mlrun.api.schemas.BackgroundTaskState.succeeded
    background_schema.metadata.updated = datetime.datetime.utcnow()
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    aioresponses_mock.get(
        f"{api_url}/api/v1/background-tasks/{task_name}", payload=response_body
    )
    response = await chief_client.get_internal_background_task(task_name)
    assert response.status_code == http.HTTPStatus.OK
    background_task = _transform_response_to_background_task(response)
    assert background_task.metadata.name == task_name
    assert (
        background_task.status.state == mlrun.api.schemas.BackgroundTaskState.succeeded
    )
    assert background_task.metadata.created == background_schema.metadata.created
    assert background_task.metadata.updated == background_schema.metadata.updated
    assert background_task.metadata.updated > background_task.metadata.created


@pytest.mark.asyncio
async def test_get_background_task_from_chief_failed(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    aioresponses_mock: aioresponses_mock,
):
    task_name = "test-for-chief"
    aioresponses_mock.get(
        f"{api_url}/api/v1/background-tasks/{task_name}",
        status=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
    )
    response = await chief_client.get_internal_background_task(task_name)
    assert response.status_code == http.HTTPStatus.INTERNAL_SERVER_ERROR.value

    # request did not retry
    aioresponses_mock.assert_called_once()


@pytest.mark.asyncio
async def test_retry_on_exception(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    aioresponses_mock: aioresponses_mock,
):

    # ensure the session to make sure the retry options are set
    await chief_client._ensure_session()
    retry_attempts = chief_client._session.retry_options.attempts

    task_name = "test-for-chief"
    for i in range(retry_attempts):
        aioresponses_mock.get(
            f"{api_url}/api/v1/background-tasks/{task_name}",
            exception=ClientConnectorError(
                unittest.mock.MagicMock(
                    code=500,
                ),
                ConnectionResetError(),
            ),
        )
    with pytest.raises(aiohttp.ClientConnectionError):
        await chief_client.get_internal_background_task(task_name)

    assert aioresponses_mock.called_times() == retry_attempts, "request did not retry"


@pytest.mark.asyncio
async def test_trigger_migration_succeeded(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    aioresponses_mock: aioresponses_mock,
):
    task_name = "test-for-chief"
    background_schema = _generate_background_task(task_name)
    # using jsonable_encoder because datetime isn't json serializable object
    # https://fastapi.tiangolo.com/tutorial/encoder/
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    aioresponses_mock.post(
        f"{api_url}/api/v1/operations/migrations",
        payload=response_body,
        status=http.HTTPStatus.ACCEPTED,
    )
    response = await chief_client.trigger_migrations()
    assert response.status_code == http.HTTPStatus.ACCEPTED
    background_task = _transform_response_to_background_task(response)
    assert background_task.metadata.name == task_name
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    assert background_task.metadata.created == background_schema.metadata.created

    background_schema.status.state = mlrun.api.schemas.BackgroundTaskState.succeeded
    background_schema.metadata.updated = datetime.datetime.utcnow()
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    aioresponses_mock.post(
        f"{api_url}/api/v1/operations/migrations",
        payload=response_body,
        status=http.HTTPStatus.ACCEPTED,
    )
    response = await chief_client.trigger_migrations()
    assert response.status_code == http.HTTPStatus.ACCEPTED
    background_task = _transform_response_to_background_task(response)
    assert background_task.metadata.name == task_name
    assert (
        background_task.status.state == mlrun.api.schemas.BackgroundTaskState.succeeded
    )
    assert background_task.metadata.created == background_schema.metadata.created
    assert background_task.metadata.updated == background_schema.metadata.updated
    assert background_task.metadata.updated > background_task.metadata.created


@pytest.mark.asyncio
async def test_trigger_migrations_from_chief_failures(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    aioresponses_mock: aioresponses_mock,
):
    aioresponses_mock.post(
        f"{api_url}/api/v1/operations/migrations",
        status=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
    )
    response = await chief_client.trigger_migrations()
    assert response.status_code == http.HTTPStatus.INTERNAL_SERVER_ERROR.value
    assert not response.body

    aioresponses_mock.post(
        f"{api_url}/api/v1/operations/migrations",
        status=http.HTTPStatus.PRECONDITION_FAILED.value,
        body="Migrations were already triggered and failed. Restart the API to retry",
    )
    response = await chief_client.trigger_migrations()
    assert response.status_code == http.HTTPStatus.PRECONDITION_FAILED.value
    assert "Migrations were already triggered and failed" in response.body.decode(
        "utf-8"
    )


@pytest.mark.asyncio
async def test_trigger_migrations_chief_restarted_while_executing_migrations(
    api_url: str,
    chief_client: mlrun.api.utils.clients.chief.Client,
    aioresponses_mock: aioresponses_mock,
):
    task_name = "test-bg-failed"

    background_schema = _generate_background_task(task_name)
    # using jsonable_encoder because datetime isn't json serializable object
    # https://fastapi.tiangolo.com/tutorial/encoder/
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    aioresponses_mock.post(
        f"{api_url}/api/v1/operations/migrations",
        payload=response_body,
        status=http.HTTPStatus.ACCEPTED,
    )
    response = await chief_client.trigger_migrations()
    assert response.status_code == http.HTTPStatus.ACCEPTED
    background_task = _transform_response_to_background_task(response)
    assert background_task.metadata.name == task_name
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    assert background_task.metadata.created == background_schema.metadata.created

    # in internal background tasks, failed state is only when the background task doesn't exists in memory,
    # which means the api was restarted
    background_schema.status.state = mlrun.api.schemas.BackgroundTaskState.failed
    response_body = fastapi.encoders.jsonable_encoder(background_schema)
    aioresponses_mock.get(
        f"{api_url}/api/v1/background-tasks/{task_name}", payload=response_body
    )
    response = await chief_client.get_internal_background_task(task_name)
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


@pytest.mark.parametrize(
    "session_cookie, expected_cookie_header",
    [
        # no escape is needed, sanity
        ("j", "j"),
        # escape is needed
        ('j:{"', "j%3A%7B%22"),
        # do not double escape
        ("j%3A%7B%22", "j%3A%7B%22"),
    ],
)
@pytest.mark.asyncio
async def test_do_not_escape_cookie(
    chief_client, session_cookie, expected_cookie_header
):
    async def handler(request):
        assert (
            request.headers["cookie"] == f"session={expected_cookie_header}"
        ), "Cookie header escaping is malfunctioning"
        assert (
            request.cookies["session"] == expected_cookie_header
        ), "Cookie session escaping is malfunctioning"
        return aiohttp.web.Response(status=200)

    mock_request = fastapi.Request({"type": "http"})
    mock_request._headers = starlette.datastructures.Headers()
    mock_request._cookies = {"session": session_cookie}
    mock_request._query_params = starlette.datastructures.QueryParams()

    app = aiohttp.web.Application()
    app.router.add_post("/api/v1/operations/migrations", handler)
    async with TestClient(TestServer(app)) as client:
        chief_client._api_url = ""
        await chief_client._ensure_session()
        chief_client._session._client = client

        # set that to make sure session escaping is on
        # coupled with chief_client._resolve_request_kwargs_from_request logic.
        mlrun.mlconf.igz_version = "0.5.0"
        response = await chief_client.trigger_migrations(mock_request)
        assert response.status_code == 200
