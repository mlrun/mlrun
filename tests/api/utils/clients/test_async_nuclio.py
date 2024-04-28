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

import pytest
from aioresponses import aioresponses as aioresponses_

import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.runtimes.nuclio.api_gateway
import server.api.utils.clients.async_nuclio


@pytest.fixture()
async def api_url() -> str:
    return "http://nuclio-dashboard-url"


@pytest.fixture()
def nuclio_client(
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
async def test_nuclio_get_api_gateway(
    api_url,
    nuclio_client,
    mock_aioresponse,
):
    api_gateway = mlrun.runtimes.nuclio.api_gateway.APIGateway(
        metadata=mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(
            name="test-basic",
        ),
        spec=mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
            functions=["test"],
            project="default-project",
        ),
    )
    api_gateway.with_basic_auth("test", "test")
    api_gateway.with_canary(["test", "test2"], [20, 80])

    request_url = f"{api_url}/api/api_gateways/test-basic"
    mock_aioresponse.get(
        request_url,
        payload=api_gateway.to_scheme().dict(),
        status=http.HTTPStatus.ACCEPTED,
    )
    r = await nuclio_client.get_api_gateway("test-basic", "default")
    received_api_gateway = mlrun.runtimes.nuclio.api_gateway.APIGateway.from_scheme(r)
    assert received_api_gateway.name == api_gateway.metadata.name
    assert received_api_gateway.description == api_gateway.spec.description
    assert (
        received_api_gateway.authentication.authentication_mode
        == api_gateway.spec.authentication.authentication_mode
    )
    assert received_api_gateway.spec.functions == ["test", "test2"]
    assert received_api_gateway.spec.canary == [20, 80]


@pytest.mark.asyncio
async def test_nuclio_delete_api_gateway(
    api_url,
    nuclio_client,
    mock_aioresponse,
):
    request_url = f"{api_url}/api/api_gateways/"
    mock_aioresponse.delete(
        request_url,
        payload={"metadata": {"name": "test-basic"}},
        status=http.HTTPStatus.NO_CONTENT,
    )
    await nuclio_client.delete_api_gateway("test-basic", "default")


@pytest.mark.asyncio
async def test_nuclio_store_api_gateway(
    api_url,
    nuclio_client,
    mock_aioresponse,
):
    request_url = f"{api_url}/api/api_gateways/new-gw"
    api_gateway = mlrun.runtimes.nuclio.api_gateway.APIGateway(
        metadata=mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(
            name="new-gw",
        ),
        spec=mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
            functions=["test-func"],
            project="default",
        ),
    )

    mock_aioresponse.put(
        request_url,
        status=http.HTTPStatus.ACCEPTED,
        payload=mlrun.common.schemas.APIGateway(
            metadata=mlrun.common.schemas.APIGatewayMetadata(
                name="new-gw",
            ),
            spec=mlrun.common.schemas.APIGatewaySpec(
                name="new-gw",
                path="/",
                host="test.host",
                upstreams=[
                    mlrun.common.schemas.APIGatewayUpstream(
                        nucliofunction={"name": "test-func"}
                    )
                ],
            ),
        ).dict(),
    )
    await nuclio_client.store_api_gateway(
        project_name="default",
        api_gateway_name="new-gw",
        api_gateway=api_gateway.to_scheme(),
    )


@pytest.mark.asyncio
async def test_nuclio_delete_function(
    api_url,
    nuclio_client,
    mock_aioresponse,
):
    request_url = f"{api_url}/api/functions/"
    mock_aioresponse.delete(
        request_url,
        payload={"metadata": {"name": "test-basic"}},
        status=http.HTTPStatus.NO_CONTENT,
    )
    await nuclio_client.delete_function("test-basic", "default")
