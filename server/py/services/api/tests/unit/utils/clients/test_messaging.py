# Copyright 2024 Iguazio
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
import http
import threading
import time
import unittest.mock

import aioresponses
import fastapi
import pytest
import requests_mock as requests_mock_package

import mlrun.errors
from tests.common_fixtures import aioresponses_mock

import framework.utils.clients.discovery
import framework.utils.clients.messaging


@pytest.fixture
def fastapi_request():
    fastapi_app = unittest.mock.Mock()
    fastapi_app.extra = {"mlrun_service_name": "test"}
    return fastapi.Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/proxy-service/success",
            "headers": [(b"host", b"http://some-other-svc/proxy-service/success")],
            "query_string": "",
            "state": {"request_id": "test"},
            "app": fastapi_app,
        }
    )


async def test_messaging_client_forward_request(
    aioresponses_mock: aioresponses_mock, fastapi_request
):
    base_url = "http://test"
    messaging_client = framework.utils.clients.messaging.Client()
    messaging_client._discovery.resolve_service_by_request = unittest.mock.Mock(
        return_value=framework.utils.clients.discovery.ServiceInstance(
            name="success-service", url=base_url
        )
    )
    aioresponses_mock.get(
        "http://test/success-service/v1/success",
        status=http.HTTPStatus.OK,
    )
    response = await messaging_client.proxy_request(fastapi_request)
    assert response.status_code == http.HTTPStatus.OK


def test_sync_delete_request(
    requests_mock: requests_mock_package.Mocker,
):
    base_url = "http://test"
    messaging_client = framework.utils.clients.messaging.Client()
    messaging_client._discovery.resolve_service_by_request = unittest.mock.Mock(
        return_value=framework.utils.clients.discovery.ServiceInstance(
            name="success-service", url=base_url
        )
    )

    api_url = "http://test/success-service/v1"
    requests_mock.delete(f"{api_url}/resource", status_code=http.HTTPStatus.NO_CONTENT)
    response = messaging_client.delete(
        path="/resource", headers={"authorization": "Bearer test"}
    )
    assert response.status_code == http.HTTPStatus.NO_CONTENT

    response = messaging_client.delete(
        path="resource", headers={"authorization": "Bearer test"}
    )
    assert response.status_code == http.HTTPStatus.NO_CONTENT

    error_message = "Resource not found"
    requests_mock.delete(
        f"{api_url}/not-a-resource",
        status_code=http.HTTPStatus.NOT_FOUND.value,
        json={
            "errors": [
                {"status": http.HTTPStatus.NOT_FOUND.value, "detail": error_message}
            ]
        },
    )

    response = messaging_client.delete(path="not-a-resource", raise_on_failure=False)
    assert response.status_code == http.HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["detail"] == error_message

    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        messaging_client.delete(path="not-a-resource")


async def test_messaging_client_forward_request_with_body(
    aioresponses_mock: aioresponses_mock,
):
    base_url = "http://test"
    messaging_client = framework.utils.clients.messaging.Client()
    messaging_client._discovery.resolve_service_by_request = unittest.mock.Mock(
        return_value=framework.utils.clients.discovery.ServiceInstance(
            name="success-service", url=base_url
        )
    )

    def _f(*args, **kwargs):
        assert kwargs["headers"].get("authorization") == "Bearer test"
        return aioresponses.CallbackResult(
            status=http.HTTPStatus.CREATED.value,
            payload={"body": "success"},
        )

    aioresponses_mock.post(
        "http://test/success-service/v1/success?x=1&x=2",
        callback=_f,
    )
    fastapi_app = unittest.mock.Mock()
    future = asyncio.Future()
    future.set_result(
        {
            "type": "http.request",
            "body": b"1",
        }
    )
    _receive = unittest.mock.Mock(return_value=future)
    request = fastapi.Request(
        receive=_receive,
        scope={
            "type": "http",
            "method": "POST",
            "path": "/proxy-service/success",
            "headers": [
                (b"host", b"http://some-other-svc/proxy-service/success"),
                (b"content-length", b"1"),
                (b"authorization", b"Bearer test"),
            ],
            "query_string": b"x=1&x=2",
            # Below are mandatory fields, although they are irrelevant for the test
            "state": {"request_id": "test"},
            "app": fastapi_app,
        },
    )
    response = await messaging_client.proxy_request(request)
    decoded_body = str(response.body.decode("utf-8"))
    assert decoded_body == '{"body": "success"}'
    assert response.status_code == http.HTTPStatus.CREATED
    _receive.assert_called_once()


def test_messaging_client_is_forwarded_request(
    aioresponses_mock: aioresponses_mock, fastapi_request
):
    base_url = "http://test"
    messaging_client = framework.utils.clients.messaging.Client()
    messaging_client._discovery.resolve_service_by_request = unittest.mock.Mock(
        return_value=framework.utils.clients.discovery.ServiceInstance(
            name="success-service", url=base_url
        )
    )
    assert messaging_client.is_forwarded_request(fastapi_request) is True


def test_messaging_client_should_not_forward_request(
    aioresponses_mock: aioresponses_mock, fastapi_request
):
    messaging_client = framework.utils.clients.messaging.Client()
    messaging_client._discovery.resolve_service_by_request = unittest.mock.Mock(
        return_value=None
    )
    assert messaging_client.is_forwarded_request(fastapi_request) is False


def test_sessions_are_different_per_thread():
    num_threads = 3
    session_ids = [None] * num_threads
    sessions = [None] * num_threads
    threads = []

    def thread_worker(index):
        async def get_session():
            client = framework.utils.clients.messaging.Client()
            session = await client._resolve_session()
            session_ids[index] = id(session) if session else None
            sessions[index] = session
            # sleep to ensure multiple threads remain active simultaneously
            time.sleep(2)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # close the session before closing the loop
        loop.run_until_complete(get_session())
        sessions[index].close()
        loop.close()

    for i in range(num_threads):
        t = threading.Thread(target=thread_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Ensure all session IDs are unique per thread
    assert None not in session_ids, f"Some sessions were not initialized: {session_ids}"
    assert (
        len(set(session_ids)) == num_threads
    ), f"Sessions should be unique per thread, got: {session_ids}"
