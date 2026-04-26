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

import unittest
from unittest.mock import patch

import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun
import mlrun.common.schemas
import mlrun.runtimes.nuclio
from mlrun.common.constants import MLRUN_FUNCTIONS_ANNOTATION
from mlrun.common.types import AuthenticationMode

import framework.utils.clients.async_nuclio
import framework.utils.clients.iguazio.v3
import services.api.crud
import services.api.tests.unit.api.utils

PROJECT = "project-name"


async def test_deploy_function(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
):
    # ensure the project exists
    services.api.tests.unit.api.utils.create_project(client, PROJECT)
    func_name = "test"

    # mock the actual function deployment as it is not relevant for this test
    with patch("services.api.api.endpoints.nuclio._deploy_function") as f:
        f.return_value = mlrun.runtimes.RemoteRuntime()
        response = client.post(
            f"projects/{PROJECT}/nuclio/{func_name}/deploy",
            json={
                "function": {},
            },
        )
        f.assert_called_once()
        assert response.status_code == 200


@patch.object(framework.utils.clients.async_nuclio.Client, "list_api_gateways")
def test_list_api_gateways(
    list_api_gateway_mocked, client: fastapi.testclient.TestClient
):
    mlrun.mlconf.httpdb.authentication.mode = AuthenticationMode.IGUAZIO
    framework.utils.clients.iguazio.v3.AsyncClient().verify_request_session = (
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
                    "authenticationMode": "none",
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


@patch.object(framework.utils.clients.async_nuclio.Client, "get_api_gateway")
@patch.object(framework.utils.clients.async_nuclio.Client, "api_gateway_exists")
@patch.object(framework.utils.clients.async_nuclio.Client, "store_api_gateway")
@patch.object(services.api.crud.Functions, "add_function_external_invocation_url")
def test_store_api_gateway(
    add_function_external_invocation_url_mocked,
    store_api_gateway_mocked,
    api_gateway_exists_mocked,
    get_api_gateway_mocked,
    client: fastapi.testclient.TestClient,
):
    mlrun.mlconf.httpdb.authentication.mode = AuthenticationMode.IGUAZIO
    framework.utils.clients.iguazio.v3.AsyncClient().verify_request_session = (
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


@pytest.mark.parametrize(
    "async_spec, expected_mode, expected_async_struct, expected_workers",
    [
        # None case - sync mode with default workers
        (None, "sync", None, 8),
        # Async enabled with default max_connections
        (
            mlrun.runtimes.nuclio.function.AsyncSpec(enabled=True),
            "async",
            {"maxConnectionsNumber": None, "connectionAvailabilityTimeout": None},
            1,
        ),
        # Async enabled with custom settings
        (
            mlrun.runtimes.nuclio.function.AsyncSpec(
                enabled=True, max_connections=500, connection_availability_timeout=30
            ),
            "async",
            {"maxConnectionsNumber": 500, "connectionAvailabilityTimeout": 30},
            1,
        ),
        # Async explicitly disabled
        (
            mlrun.runtimes.nuclio.function.AsyncSpec(enabled=False),
            "sync",
            None,
            8,
        ),
    ],
)
@pytest.mark.parametrize("nuclio_support_async", [True, False])
def test_with_http_async_spec(
    async_spec,
    expected_mode,
    expected_async_struct,
    expected_workers,
    nuclio_support_async,
):
    """Test with_http method with various async_spec configurations."""
    func = mlrun.runtimes.nuclio.function.RemoteRuntime()

    with patch(
        "mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility",
        return_value=nuclio_support_async,
    ):
        if not nuclio_support_async and async_spec is not None:
            with pytest.raises(
                mlrun.errors.MLRunValueError,
                match="Async spec is only supported from Nuclio 1.15.3",
            ):
                func.with_http(async_spec=async_spec)
        else:
            func.with_http(async_spec=async_spec)
            trigger = func.spec.config.get("spec.triggers.http")
            assert trigger is not None
            if nuclio_support_async:
                # Check mode
                assert trigger.get("mode") == expected_mode

                # Check workers
                assert trigger.get("maxWorkers") == expected_workers
            else:
                assert trigger.get("mode") is None
                assert (
                    trigger.get("maxWorkers") == 8
                )  # Default workers when async is not supported

            # Check async struct
            if expected_async_struct is not None:
                assert trigger.get("async") == expected_async_struct
            else:
                assert "async" not in trigger


def test_same_invoke_url_skips_full_deletion():
    """
    When the existing and new API gateway have the same invoke URL,
    _delete_functions_external_invocation_url should NOT be called
    for ALL functions.  It may still be called for individually
    removed functions (the elif branch), but not the full-URL-change
    branch.
    """
    existing = _make_api_gateway("https://my.host", "/api", ["proj/fn1", "proj/fn2"])
    updated = _make_api_gateway("https://my.host", "/api", ["proj/fn1", "proj/fn2"])

    # Sanity: both return the same URL string
    assert existing.get_invoke_url() == updated.get_invoke_url()

    # But comparing the methods themselves (the old buggy code) would
    # always be True because they are different bound-method objects
    assert existing.get_invoke_url != updated.get_invoke_url  # methods differ

    # The fixed comparison (calling the methods) should be False
    assert not (existing.get_invoke_url() != updated.get_invoke_url()), (
        "Calling get_invoke_url() should show equal URLs"
    )


def test_different_invoke_url_triggers_deletion():
    """
    When the invoke URL really changed, the comparison should detect it.
    """
    existing = _make_api_gateway("https://old.host", "/api", ["proj/fn1"])
    updated = _make_api_gateway("https://new.host", "/api", ["proj/fn1"])

    # The URLs are different so the condition should be True
    assert existing.get_invoke_url() != updated.get_invoke_url()


def test_unused_functions_detected_when_url_unchanged():
    """
    When the URL is the same but functions were removed, the elif branch
    should be reachable (it was unreachable before the fix).
    """
    existing = _make_api_gateway("https://my.host", "/api", ["proj/fn1", "proj/fn2"])
    updated = _make_api_gateway("https://my.host", "/api", ["proj/fn1"])

    # URL is the same
    assert existing.get_invoke_url() == updated.get_invoke_url()

    # fn2 was removed
    unused_functions = [
        func
        for func in existing.get_function_names()
        if func not in updated.get_function_names()
    ]
    assert unused_functions == ["proj/fn2"]


def _make_api_gateway(host: str, path: str, function_names: list[str]):
    """Build a minimal APIGateway schema with the given host, path and functions."""
    upstreams = [
        mlrun.common.schemas.APIGatewayUpstream(
            nucliofunction={"name": name},
            percentage=100 // len(function_names),
        )
        for name in function_names
    ]
    return mlrun.common.schemas.APIGateway(
        metadata=mlrun.common.schemas.APIGatewayMetadata(name="gw"),
        spec=mlrun.common.schemas.APIGatewaySpec(
            name="gw",
            host=host,
            path=path,
            upstreams=upstreams,
        ),
    )
