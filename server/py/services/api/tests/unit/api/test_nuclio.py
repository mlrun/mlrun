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
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import fastapi
import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun
import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes
import mlrun.runtimes.nuclio
from mlrun.common.constants import MLRUN_FUNCTIONS_ANNOTATION
from mlrun.common.types import AuthenticationMode

import framework.utils.clients.async_nuclio
import framework.utils.clients.iguazio.v3
import services.api.crud
import services.api.crud.runtimes.nuclio.function
import services.api.tests.unit.api.utils
from services.api.api.endpoints.nuclio import _deploy_function, _deploy_nuclio_runtime

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


def test_deploy_function_failure_marks_function_errored(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
):
    """A deploy failure before Nuclio is reached must mark the persisted function
    errored, since the deploy-status poller never runs to correct it."""
    services.api.tests.unit.api.utils.create_project(client, PROJECT)
    func_name = "test-app"

    # Simulate the build phase having already persisted the function as ready.
    fn = mlrun.new_function(name=func_name, kind="application", project=PROJECT)
    fn.status.state = mlrun.common.schemas.FunctionState.ready
    services.api.crud.Functions().store_function(
        db, fn.to_dict(), func_name, PROJECT, tag="latest", versioned=False
    )

    with patch(
        "services.api.launcher.ServerSideLauncher.enrich_runtime",
        side_effect=mlrun.errors.MLRunNotFoundError("code artifact not found"),
    ):
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _deploy_function(
                db_session=db,
                auth_info=mlrun.common.schemas.AuthInfo(),
                project=PROJECT,
                name=func_name,
                function={
                    "kind": "application",
                    "metadata": {"name": func_name, "project": PROJECT},
                },
                builder_env=None,
                client_version=None,
                client_python_version=None,
            )

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
    stored = services.api.crud.Functions().get_function(
        db, func_name, PROJECT, tag="latest"
    )
    assert stored["status"]["state"] == mlrun.common.schemas.FunctionState.error


def test_deploy_function_failure_status_update_error_is_swallowed(
    db: sqlalchemy.orm.Session,
):
    """A failure while marking the function errored must not mask the original
    deploy error - the original failure is still raised."""
    with (
        patch(
            "services.api.launcher.ServerSideLauncher.enrich_runtime",
            side_effect=mlrun.errors.MLRunNotFoundError("code artifact not found"),
        ),
        patch(
            "services.api.crud.Functions.update_function",
            side_effect=Exception("db unavailable"),
        ),
    ):
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _deploy_function(
                db_session=db,
                auth_info=mlrun.common.schemas.AuthInfo(),
                project=PROJECT,
                name="test-app",
                function={
                    "kind": "application",
                    "metadata": {"name": "test-app", "project": PROJECT},
                },
                builder_env=None,
                client_version=None,
                client_python_version=None,
            )

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
    assert "code artifact not found" in str(exc_info.value.detail)


def test_deploy_function_after_nuclio_submission_does_not_mark_errored(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
):
    """Once the deploy was submitted to Nuclio, a later failure must not override
    the function state - a Nuclio resource may exist and the poller reconciles it."""
    services.api.tests.unit.api.utils.create_project(client, PROJECT)
    func_name = "test-app"

    fn = mlrun.new_function(name=func_name, kind="application", project=PROJECT)
    fn.status.state = mlrun.common.schemas.FunctionState.ready
    services.api.crud.Functions().store_function(
        db, fn.to_dict(), func_name, PROJECT, tag="latest", versioned=False
    )

    def _submit_then_fail(*args, on_submit=None, **kwargs):
        # the request reached Nuclio, then the response was lost
        on_submit()
        raise mlrun.errors.MLRunRuntimeError("connection to nuclio lost")

    with (
        patch("services.api.launcher.ServerSideLauncher.enrich_runtime"),
        patch("framework.api.utils.apply_enrichment_and_validation_on_function"),
        patch(
            "mlrun.runtimes.nuclio.application.application.ApplicationRuntime.pre_deploy_validation"
        ),
        patch(
            "services.api.crud.runtimes.nuclio.function.deploy_nuclio_function",
            side_effect=_submit_then_fail,
        ),
    ):
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _deploy_function(
                db_session=db,
                auth_info=mlrun.common.schemas.AuthInfo(),
                project=PROJECT,
                name=func_name,
                function=fn.to_dict(),
                builder_env=None,
                client_version=None,
                client_python_version=None,
            )

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
    stored = services.api.crud.Functions().get_function(
        db, func_name, PROJECT, tag="latest"
    )
    assert stored["status"]["state"] == mlrun.common.schemas.FunctionState.ready


def test_deploy_nuclio_function_invokes_on_submit_before_submission():
    """deploy_nuclio_function must call on_submit immediately before submitting the
    config to Nuclio, so callers can tell a resource may exist."""
    fn = mlrun.new_function(name="test-fn", kind="remote", project=PROJECT)
    events = []

    with (
        patch(
            "services.api.crud.runtimes.nuclio.function._compile_function_config",
            return_value=("test-fn", PROJECT, {}),
        ),
        patch("services.api.crud.runtimes.nuclio.helpers.enrich_function_with_ingress"),
        patch(
            "nuclio.deploy.deploy_config",
            side_effect=lambda *args, **kwargs: events.append("submit"),
        ),
    ):
        services.api.crud.runtimes.nuclio.function.deploy_nuclio_function(
            fn,
            auth_info=mlrun.common.schemas.AuthInfo(),
            on_submit=lambda: events.append("on_submit"),
        )

    assert events == ["on_submit", "submit"]


def test_deploy_function_success_does_not_mark_errored(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
):
    """A successful deploy returns the function and never marks it errored."""
    services.api.tests.unit.api.utils.create_project(client, PROJECT)
    func_name = "test-app"
    fn = mlrun.new_function(name=func_name, kind="application", project=PROJECT)

    def _succeed(*args, on_submit=None, **kwargs):
        on_submit()
        return args[5]

    with (
        patch("services.api.launcher.ServerSideLauncher.enrich_runtime"),
        patch("framework.api.utils.apply_enrichment_and_validation_on_function"),
        patch(
            "mlrun.runtimes.nuclio.application.application.ApplicationRuntime.pre_deploy_validation"
        ),
        patch(
            "services.api.api.endpoints.nuclio._deploy_nuclio_runtime",
            side_effect=_succeed,
        ),
        patch(
            "services.api.api.endpoints.nuclio._mark_function_deploy_error"
        ) as mock_mark,
    ):
        result = _deploy_function(
            db_session=db,
            auth_info=mlrun.common.schemas.AuthInfo(),
            project=PROJECT,
            name=func_name,
            function=fn.to_dict(),
            builder_env=None,
            client_version=None,
            client_python_version=None,
        )

    assert result is not None
    mock_mark.assert_not_called()


@pytest.mark.parametrize(
    "kind",
    [mlrun.runtimes.RuntimeKinds.remote, mlrun.runtimes.RuntimeKinds.application],
)
def test_nuclio_app_track_models_raises_on_missing_credentials(kind):
    """_deploy_nuclio_runtime must reject remote/application functions with track_models when credentials are unset."""
    fn = mlrun.new_function(name="test-fn", kind=kind, project=PROJECT)
    fn.spec.track_models = True

    with (
        patch(
            "services.api.api.endpoints.nuclio.process_model_monitoring_secret",
            return_value=None,
        ),
        patch(
            "services.api.crud.model_monitoring.deployment.MonitoringDeployment.check_if_credentials_are_set",
            side_effect=mlrun.errors.MLRunBadRequestError("credentials not set"),
        ),
    ):
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _deploy_nuclio_runtime(
                auth_info=mlrun.common.schemas.AuthInfo(),
                builder_env=None,
                client_python_version=None,
                client_version=None,
                db_session=MagicMock(),
                fn=fn,
            )
    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value


@pytest.mark.parametrize(
    "kind",
    [mlrun.runtimes.RuntimeKinds.remote, mlrun.runtimes.RuntimeKinds.application],
)
def test_nuclio_app_track_models_calls_credential_check(kind):
    """_deploy_nuclio_runtime must call check_if_credentials_are_set for remote/application with track_models."""
    fn = mlrun.new_function(name="test-fn", kind=kind, project=PROJECT)
    fn.spec.track_models = True

    with (
        patch(
            "services.api.api.endpoints.nuclio.process_model_monitoring_secret",
            return_value=None,
        ),
        patch(
            "services.api.crud.model_monitoring.deployment.MonitoringDeployment.check_if_credentials_are_set"
        ) as mock_check,
        patch("services.api.crud.runtimes.nuclio.function.deploy_nuclio_function"),
    ):
        _deploy_nuclio_runtime(
            auth_info=mlrun.common.schemas.AuthInfo(),
            builder_env=None,
            client_python_version=None,
            client_version=None,
            db_session=MagicMock(),
            fn=fn,
        )

    mock_check.assert_called_once()
