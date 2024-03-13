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
import server.py.services.api.utils.clients.async_nuclio
import server.py.services.api.utils.clients.iguazio

PROJECT = "project-name"


@patch.object(
    server.py.services.api.utils.clients.async_nuclio.Client, "list_api_gateways"
)
def test_list_api_gateways(
    list_api_gateway_mocked, client: fastapi.testclient.TestClient
):
    mlrun.mlconf.httpdb.authentication.mode = "iguazio"
    server.py.services.api.utils.clients.iguazio.AsyncClient().verify_request_session = unittest.mock.AsyncMock(
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
    nuclio_api_response_body = {
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

    expected_response_body = list(nuclio_api_response_body.values())

    list_api_gateway_mocked.return_value = nuclio_api_response_body
    response = client.get(
        f"projects/{PROJECT}/nuclio/api-gateways",
    )

    assert response.json() == expected_response_body
