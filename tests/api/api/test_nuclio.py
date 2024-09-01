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
import pytest

import mlrun
import mlrun.common.schemas
import mlrun.runtimes.nuclio
import server.api.crud
import server.api.utils.clients.async_nuclio
import server.api.utils.clients.iguazio
from mlrun.common.constants import MLRUN_FUNCTIONS_ANNOTATION

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
                "metadata": {"name": "new-gw", "labels": {}, "annotations": {}},
                "spec": {
                    "name": "new-gw",
                    "path": "/",
                    "authentication_mode": "none",
                    "upstreams": [
                        {
                            "kind": "nucliofunction",
                            "nucliofunction": {"name": "test-func"},
                            "percentage": 0,
                            "port": 0,
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
@patch.object(server.api.crud.Functions, "add_function_external_invocation_url")
def test_store_api_gateway(
    add_function_external_invocation_url_mocked,
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
    add_function_external_invocation_url_mocked.return_value = True
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


@pytest.mark.parametrize(
    "functions, expected_nuclio_function_names, expected_mlrun_functions_label",
    [
        (
            ["test-func"],
            ["test-project-test-func"],
            "test-project/test-func",
        ),
        (
            ["test-func1", "test-func2"],
            ["test-project-test-func1", "test-project-test-func2"],
            "test-project/test-func1&test-project/test-func2",
        ),
        (
            ["test-func1:latest", "test-func2:latest"],
            ["test-project-test-func1", "test-project-test-func2"],
            "test-project/test-func1:latest&test-project/test-func2:latest",
        ),
        (
            ["test-func1:tag1", "test-func2:tag2"],
            ["test-project-test-func1-tag1", "test-project-test-func2-tag2"],
            "test-project/test-func1:tag1&test-project/test-func2:tag2",
        ),
    ],
)
def test_mlrun_function_translation_to_nuclio(
    functions, expected_nuclio_function_names, expected_mlrun_functions_label
):
    project_name = "test-project"
    api_gateway_client_side = mlrun.runtimes.APIGateway(
        metadata=mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(name="new-gw"),
        spec=mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
            functions=functions, project=project_name
        ),
    )
    api_gateway_server_side = api_gateway_client_side.to_scheme().enrich_mlrun_names()
    assert (
        api_gateway_server_side.get_function_names() == expected_nuclio_function_names
    )

    assert (
        api_gateway_server_side.metadata.annotations[MLRUN_FUNCTIONS_ANNOTATION]
        == expected_mlrun_functions_label
    )
    api_gateway_with_replaced_nuclio_names_to_mlrun = (
        api_gateway_server_side.replace_nuclio_names_with_mlrun_names()
    )
    assert (
        api_gateway_with_replaced_nuclio_names_to_mlrun.get_function_names()
        == api_gateway_client_side.spec.functions
    )
