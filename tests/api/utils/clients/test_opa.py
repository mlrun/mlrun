import http

import deepdiff
import pytest
import requests_mock as requests_mock_package

import mlrun.api.schemas
import mlrun.api.utils.clients.opa
import mlrun.config
import mlrun.errors


@pytest.fixture()
async def api_url() -> str:
    api_url = "http://127.0.0.1:8181"
    mlrun.mlconf.httpdb.authorization.opa.address = api_url
    return api_url


@pytest.fixture()
async def permission_query_path() -> str:
    permission_query_path = "/v1/data/service/authz/allow"
    mlrun.mlconf.httpdb.authorization.opa.permission_query_path = permission_query_path
    return permission_query_path


@pytest.fixture()
async def opa_client(api_url: str, permission_query_path: str,) -> mlrun.api.utils.clients.opa.Client:
    mlrun.mlconf.httpdb.authorization.opa.log_level = 10
    mlrun.mlconf.httpdb.authorization.mode = "opa"
    client = mlrun.api.utils.clients.opa.Client()
    # force running init again so the configured api url will be used
    client.__init__()
    return client


def test_query_permissions_success(
        api_url: str,
        permission_query_path: str,
        opa_client: mlrun.api.utils.clients.opa.Client,
        requests_mock: requests_mock_package.Mocker,
):
    resource = '/projects/project-name/functions/function-name'
    action = mlrun.api.schemas.AuthorizationAction.create
    auth_info = mlrun.api.schemas.AuthInfo(
        user_id="user-id",
        user_group_ids=["user-group-id-1", "user-group-id-2"]
    )

    def mock_permission_query_success(request, context):
        assert (
                deepdiff.DeepDiff(
                    opa_client._generate_permission_request_body(resource, action.value, auth_info),
                    request.json(),
                    ignore_order=True,
                )
                == {}
        )
        context.status_code = http.HTTPStatus.OK.value
        return {
            "result": True
        }

    requests_mock.post(f"{api_url}{permission_query_path}", json=mock_permission_query_success)
    allowed = opa_client.query_permissions(
        resource,
        action,
        auth_info
    )
    assert allowed is True


def test_query_permissions_failure(
        api_url: str,
        permission_query_path: str,
        opa_client: mlrun.api.utils.clients.opa.Client,
        requests_mock: requests_mock_package.Mocker,
):
    resource = '/projects/project-name/functions/function-name'
    action = mlrun.api.schemas.AuthorizationAction.create
    auth_info = mlrun.api.schemas.AuthInfo(
        user_id="user-id",
        user_group_ids=["user-group-id-1", "user-group-id-2"]
    )

    def mock_permission_query_failure(request, context):
        assert (
                deepdiff.DeepDiff(
                    opa_client._generate_permission_request_body(resource, action.value, auth_info),
                    request.json(),
                    ignore_order=True,
                )
                == {}
        )
        context.status_code = http.HTTPStatus.OK.value
        return {
            "result": False
        }
    requests_mock.post(f"{api_url}{permission_query_path}", json=mock_permission_query_failure)
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError, match=f"Not allowed to {action} resource {resource}"):
        opa_client.query_permissions(
            resource,
            action,
            auth_info
        )

