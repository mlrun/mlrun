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
import unittest.mock

import fastapi

from tests.common_fixtures import aioresponses_mock

import framework.utils.clients.discovery
import framework.utils.clients.messaging


async def test_messaging_client_forward_request(
    aioresponses_mock: aioresponses_mock,
):
    base_url = "http://test"
    messaging_client = framework.utils.clients.messaging.Client()
    messaging_client._discovery = unittest.mock.Mock()
    messaging_client._discovery.resolve_service_by_request = unittest.mock.Mock(
        return_value=framework.utils.clients.discovery.ServiceInstance(
            name="success-service", url=base_url
        )
    )
    fastapi_app = unittest.mock.Mock()
    fastapi_app.extra = {"mlrun_service_name": "test"}
    aioresponses_mock.get(
        "http://test/success-service/v1/success",
        status=http.HTTPStatus.OK,
    )
    request = fastapi.Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/proxy-service/success",
            "headers": [(b"host", b"http://some-other-svc/proxy-service/success")],
            # Below are mandatory fields, although they are irrelevant for the test
            "query_string": "",
            "state": {"request_id": "test"},
            "app": fastapi_app,
        },
    )
    response = await messaging_client.proxy_request(request)
    assert response.status_code == http.HTTPStatus.OK


async def test_messaging_client_forward_request_with_body(
    aioresponses_mock: aioresponses_mock,
):
    base_url = "http://test"
    messaging_client = framework.utils.clients.messaging.Client()
    messaging_client._discovery = unittest.mock.Mock()
    messaging_client._discovery.resolve_service_by_request = unittest.mock.Mock(
        return_value=framework.utils.clients.discovery.ServiceInstance(
            name="success-service", url=base_url
        )
    )
    aioresponses_mock.post(
        "http://test/success-service/v1/success",
        body="success",
        status=http.HTTPStatus.CREATED,
    )
    fastapi_app = unittest.mock.Mock()
    fastapi_app.extra = {"mlrun_service_name": "test"}
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
            ],
            # Below are mandatory fields, although they are irrelevant for the test
            "query_string": "",
            "state": {"request_id": "test"},
            "app": fastapi_app,
        },
    )
    response = await messaging_client.proxy_request(request)
    decoded_body = str(response.body.decode("utf-8"))
    assert decoded_body == "success"
    assert response.status_code == http.HTTPStatus.CREATED
    _receive.assert_called_once()
