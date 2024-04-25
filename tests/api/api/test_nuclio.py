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
import unittest
from unittest.mock import patch

import fastapi

import mlrun
import mlrun.common.schemas
import server.api.utils.clients.async_nuclio
import server.api.utils.clients.iguazio

PROJECT = "project-name"


@patch.object(server.api.utils.clients.async_nuclio.Client, "list_api_gateways")
def test_list_api_gateways(
    list_api_gateway_mocked, client: fastapi.testclient.TestClient
):
    mlrun.mlconf.httpdb.authentication.mode = "iguazio"
    server.api.utils.clients.iguazio.AsyncClient().verify_request_session = (
        unittest.mock.AsyncMock(
            return_value=(
                mlrun.common.schemas.AuthInfo(
                    username="admin",
                    session="some-session",
                    data_session="some-session",
                    user_id=None,
                    user_unix_id=0,
                    user_group_ids=[],
                )
            )
        )
    )
    nuclio_api_response_body = {
        "new-gw": mlrun.common.schemas.APIGateway(
            metadata=mlrun.common.schemas.APIGatewayMetadata(
                name="new-gw",
            ),
            spec=mlrun.common.schemas.APIGatewaySpec(
                name="new-gw",
                path="/",
                host="http://my-api-gateway.com",
                upstreams=[
                    mlrun.common.schemas.APIGatewayUpstream(
                        nucliofunction={"name": "test-func"}
                    )
                ],
            ),
        )
    }

    list_api_gateway_mocked.return_value = nuclio_api_response_body
    response = client.get(
        f"projects/{PROJECT}/api-gateways",
    )

    assert response.json() == {
        "api_gateways": {
            "new-gw": {
                "metadata": {"name": "new-gw", "labels": {}},
                "spec": {
                    "name": "new-gw",
                    "path": "/",
                    "authenticationMode": "none",
                    "upstreams": [
                        {
                            "kind": "nucliofunction",
                            "nucliofunction": {"name": "test-func"},
                            "percentage": 0,
                        }
                    ],
                    "host": "http://my-api-gateway.com",
                },
            }
        }
    }


@patch.object(server.api.utils.clients.async_nuclio.Client, "get_api_gateway")
@patch.object(server.api.utils.clients.async_nuclio.Client, "api_gateway_exists")
@patch.object(server.api.utils.clients.async_nuclio.Client, "store_api_gateway")
def test_store_api_gateway(
    store_api_gateway_mocked,
    api_gateway_exists_mocked,
    get_api_gateway_mocked,
    client: fastapi.testclient.TestClient,
):
    mlrun.mlconf.httpdb.authentication.mode = "iguazio"
    server.api.utils.clients.iguazio.AsyncClient().verify_request_session = (
        unittest.mock.AsyncMock(
            return_value=(
                mlrun.common.schemas.AuthInfo(
                    username="admin",
                    session="some-session",
                    data_session="some-session",
                    user_id=None,
                    user_unix_id=0,
                    user_group_ids=[],
                )
            )
        )
    )

    api_gateway_exists_mocked.return_value = False
    store_api_gateway_mocked.return_value = True
    get_api_gateway_mocked.return_value = mlrun.common.schemas.APIGateway(
        metadata=mlrun.common.schemas.APIGatewayMetadata(
            name="new-gw",
        ),
        spec=mlrun.common.schemas.APIGatewaySpec(
            name="new-gw",
            path="/",
            host="http://my-api-gateway.com",
            upstreams=[
                mlrun.common.schemas.APIGatewayUpstream(
                    nucliofunction={"name": "test-func"}
                )
            ],
        ),
    )

    api_gateway = mlrun.common.schemas.APIGateway(
        metadata=mlrun.common.schemas.APIGatewayMetadata(
            name="new-gw",
        ),
        spec=mlrun.common.schemas.APIGatewaySpec(
            name="new-gw",
            path="/",
            upstreams=[
                mlrun.common.schemas.APIGatewayUpstream(
                    nucliofunction={"name": "test-func"}
                )
            ],
        ),
    )

    response = client.put(
        f"projects/{PROJECT}/api-gateways/new-gw",
        json=api_gateway.dict(),
    )
    assert response.status_code == 200
