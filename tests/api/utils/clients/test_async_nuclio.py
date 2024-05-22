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
#
import http

import aiohttp
import pytest
from aioresponses import aioresponses as aioresponses_

import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import server.api.utils.clients.async_nuclio


@pytest.fixture()
async def api_url() -> str:
    return "http://nuclio-dashboard-url"


@pytest.fixture()
async def nuclio_client(
    api_url,
) -> server.api.utils.clients.async_nuclio.Client:
    auth_info = mlrun.common.schemas.AuthInfo()
    auth_info.username = "admin"
    auth_info.session = "bed854c1-c57751553"
    client = server.api.utils.clients.async_nuclio.Client(auth_info)
    client._nuclio_dashboard_url = api_url
    return client


@pytest.fixture
def mock_aioresponse():
    with aioresponses_() as m:
        yield m


@pytest.mark.asyncio
async def test_nuclio_list_api_gateways(
    api_url,
    nuclio_client,
    mock_aioresponse,
):
    response_body = {
        "test-basic": {
            "metadata": {
                "name": "test-basic",
                "namespace": "default-tenant",
                "labels": {
                    "iguazio.com/username": "admin",
                    "nuclio.io/project-name": "default",
                },
                "creationTimestamp": "2023-11-16T12:42:48Z",
            },
            "spec": {
                "host": "test-basic-default.default-tenant.app.dev62.lab.iguazeng.com",
                "name": "test-basic",
                "path": "/",
                "authenticationMode": "basicAuth",
                "authentication": {
                    "basicAuth": {"username": "test", "password": "test"}
                },
                "upstreams": [
                    {"kind": "nucliofunction", "nucliofunction": {"name": "test"}}
                ],
            },
            "status": {"name": "test-basic", "state": "ready"},
        }
    }
    request_url = f"{api_url}/api/api_gateways/"
    mock_aioresponse.get(
        request_url,
        payload=response_body,
        status=http.HTTPStatus.ACCEPTED,
    )
    r = await nuclio_client.list_api_gateways()
    assert r == response_body

    mock_aioresponse.get(request_url, status=http.HTTPStatus.UNAUTHORIZED)
    with pytest.raises(mlrun.errors.MLRunUnauthorizedError):
        await nuclio_client.list_api_gateways()
