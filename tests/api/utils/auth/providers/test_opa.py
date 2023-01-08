# Copyright 2018 Iguazio
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
import time

import aioresponses
import deepdiff
import pytest

import mlrun.api.schemas
import mlrun.api.utils.auth.providers.opa
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
async def permission_filter_path() -> str:
    permission_filter_path = "/v1/data/service/authz/filter_allowed"
    mlrun.mlconf.httpdb.authorization.opa.permission_filter_path = (
        permission_filter_path
    )
    return permission_filter_path


@pytest.fixture()
async def opa_provider(
    api_url: str,
    permission_query_path: str,
    permission_filter_path: str,
) -> mlrun.api.utils.auth.providers.opa.Provider:
    mlrun.mlconf.httpdb.authorization.opa.log_level = 10
    mlrun.mlconf.httpdb.authorization.mode = "opa"
    provider = mlrun.api.utils.auth.providers.opa.Provider()

    # force running init again so the configured api url will be used
    provider.__init__()
    yield provider

    # explicitly closing the provider's session to avoid "unclosed session" warning between tests
    if provider._session:
        await provider._session.close()


@pytest.mark.asyncio
async def test_query_permissions_success(
    api_url: str,
    permission_query_path: str,
    opa_provider: mlrun.api.utils.auth.providers.opa.Provider,
):
    resource = "/projects/project-name/functions/function-name"
    action = mlrun.api.schemas.AuthorizationAction.create
    auth_info = mlrun.api.schemas.AuthInfo(
        user_id="user-id", user_group_ids=["user-group-id-1", "user-group-id-2"]
    )

    def mock_permission_query_success(url, **kwargs):
        assert (
            deepdiff.DeepDiff(
                opa_provider._generate_permission_request_body(
                    resource, action.value, auth_info
                ),
                kwargs["json"],
                ignore_order=True,
            )
            == {}
        )
        return aioresponses.CallbackResult(
            status=http.HTTPStatus.OK.value, payload={"result": True}
        )

    with aioresponses.aioresponses() as aiohttp_mock:
        aiohttp_mock.post(
            f"{api_url}{permission_query_path}", callback=mock_permission_query_success
        )
        allowed = await opa_provider.query_permissions(resource, action, auth_info)
    assert allowed is True, "Expected query permissions to succeed"


@pytest.mark.asyncio
async def test_filter_by_permission(
    api_url: str,
    permission_filter_path: str,
    opa_provider: mlrun.api.utils.auth.providers.opa.Provider,
):
    resources = [
        {"resource_id": 1, "opa_resource": "/some-resource", "allowed": True},
        {
            "resource_id": 2,
            "opa_resource": "/two-objects-same-opa-resource",
            "allowed": True,
        },
        {
            "resource_id": 3,
            "opa_resource": "/two-objects-same-opa-resource",
            "allowed": True,
        },
        {"resource_id": 4, "opa_resource": "/not-allowed-resource", "allowed": False},
    ]
    expected_allowed_resources = [
        resource for resource in resources if resource["allowed"]
    ]
    allowed_opa_resources = [
        resource["opa_resource"] for resource in expected_allowed_resources
    ]
    action = mlrun.api.schemas.AuthorizationAction.create
    auth_info = mlrun.api.schemas.AuthInfo(
        user_id="user-id", user_group_ids=["user-group-id-1", "user-group-id-2"]
    )

    def mock_filter_query_success(url, **kwargs):
        opa_resources = [resource["opa_resource"] for resource in resources]
        assert (
            deepdiff.DeepDiff(
                opa_provider._generate_filter_request_body(
                    opa_resources, action.value, auth_info
                ),
                kwargs["json"],
                ignore_order=True,
            )
            == {}
        )
        return aioresponses.CallbackResult(
            status=http.HTTPStatus.OK.value, payload={"result": allowed_opa_resources}
        )

    with aioresponses.aioresponses() as aiohttp_mock:
        aiohttp_mock.post(
            f"{api_url}{permission_filter_path}", callback=mock_filter_query_success
        )
        allowed_resources = await opa_provider.filter_by_permissions(
            resources, lambda resource: resource["opa_resource"], action, auth_info
        )
    assert (
        deepdiff.DeepDiff(
            expected_allowed_resources,
            allowed_resources,
            ignore_order=True,
        )
        == {}
    )


@pytest.mark.asyncio
async def test_query_permissions_failure(
    api_url: str,
    permission_query_path: str,
    opa_provider: mlrun.api.utils.auth.providers.opa.Provider,
    requests_mock: aioresponses.aioresponses,
):
    resource = "/projects/project-name/functions/function-name"
    action = mlrun.api.schemas.AuthorizationAction.create
    auth_info = mlrun.api.schemas.AuthInfo(
        user_id="user-id", user_group_ids=["user-group-id-1", "user-group-id-2"]
    )

    def mock_permission_query_failure(url, **kwargs):
        assert (
            deepdiff.DeepDiff(
                opa_provider._generate_permission_request_body(
                    resource, action.value, auth_info
                ),
                kwargs["json"],
                ignore_order=True,
            )
            == {}
        )
        return aioresponses.CallbackResult(
            status=http.HTTPStatus.OK.value, payload={"result": False}
        )

    with aioresponses.aioresponses() as aiohttp_mock:
        aiohttp_mock.post(
            f"{api_url}{permission_query_path}", callback=mock_permission_query_failure
        )
        with pytest.raises(
            mlrun.errors.MLRunAccessDeniedError,
            match=f"Not allowed to {action} resource {resource}",
        ):
            await opa_provider.query_permissions(resource, action, auth_info)


@pytest.mark.asyncio
async def test_query_permissions_use_cache(
    api_url: str,
    permission_query_path: str,
    opa_provider: mlrun.api.utils.auth.providers.opa.Provider,
):
    auth_info = mlrun.api.schemas.AuthInfo(user_id="user-id")
    project_name = "project-name"
    opa_provider.add_allowed_project_for_owner(project_name, auth_info)

    with aioresponses.aioresponses() as aiohttp_mock:
        assert (
            await opa_provider.query_permissions(
                f"/projects/{project_name}/resource",
                mlrun.api.schemas.AuthorizationAction.create,
                auth_info,
            )
            is True
        )
        aiohttp_mock.assert_not_called()


def test_allowed_project_owners_cache(
    api_url: str,
    permission_query_path: str,
    opa_provider: mlrun.api.utils.auth.providers.opa.Provider,
):
    auth_info = mlrun.api.schemas.AuthInfo(user_id="user-id")
    project_name = "project-name"
    opa_provider.add_allowed_project_for_owner(project_name, auth_info)
    # ensure nothing is wrong with adding the same project twice
    opa_provider.add_allowed_project_for_owner(project_name, auth_info)
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name}/resource", auth_info
        )
        is True
    )
    assert (
        opa_provider._check_allowed_project_owners_cache(
            "/some-non-project-resource", auth_info
        )
        is False
    )
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name}/resource",
            mlrun.api.schemas.AuthInfo(user_id="other-user-id"),
        )
        is False
    )


def test_allowed_project_owners_cache_ttl_refresh(
    api_url: str,
    permission_query_path: str,
    opa_provider: mlrun.api.utils.auth.providers.opa.Provider,
):
    auth_info = mlrun.api.schemas.AuthInfo(user_id="user-id")
    opa_provider._allowed_project_owners_cache_ttl_seconds = 1
    project_name = "project-name"
    opa_provider.add_allowed_project_for_owner(project_name, auth_info)
    time.sleep(0.3)
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name}/resource", auth_info
        )
        is True
    )
    # This will refresh the ttl
    opa_provider.add_allowed_project_for_owner(project_name, auth_info)
    time.sleep(0.7)
    # by now, more than the first 1 second surely passed, so if it works, ttl refreshed
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name}/resource", auth_info
        )
        is True
    )


def test_allowed_project_owners_cache_clean_expired(
    api_url: str,
    permission_query_path: str,
    opa_provider: mlrun.api.utils.auth.providers.opa.Provider,
):
    auth_info = mlrun.api.schemas.AuthInfo(user_id="user-id")
    auth_info_2 = mlrun.api.schemas.AuthInfo(user_id="user-id-2")
    opa_provider._allowed_project_owners_cache_ttl_seconds = 2
    project_name = "project-name"
    project_name_2 = "project-name-2"
    project_name_3 = "project-name-3"
    opa_provider.add_allowed_project_for_owner(project_name, auth_info)
    time.sleep(1)
    # Note that the _check_allowed_project_owners_cache method calls the clean method so no need to call the clean
    # method explicitly
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name}/resource", auth_info
        )
        is True
    )
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name}/resource", auth_info_2
        )
        is False
    )
    opa_provider.add_allowed_project_for_owner(project_name_2, auth_info)
    opa_provider.add_allowed_project_for_owner(project_name, auth_info_2)
    time.sleep(1)
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name}/resource", auth_info
        )
        is False
    )
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name}/resource", auth_info_2
        )
        is True
    )
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name_2}/resource", auth_info
        )
        is True
    )
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name_2}/resource", auth_info_2
        )
        is False
    )
    opa_provider.add_allowed_project_for_owner(project_name_3, auth_info)
    opa_provider.add_allowed_project_for_owner(project_name_2, auth_info_2)
    time.sleep(1)
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name}/resource", auth_info
        )
        is False
    )
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name}/resource", auth_info_2
        )
        is False
    )
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name_2}/resource", auth_info
        )
        is False
    )
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name_2}/resource", auth_info_2
        )
        is True
    )
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name_3}/resource", auth_info
        )
        is True
    )
    assert (
        opa_provider._check_allowed_project_owners_cache(
            f"/projects/{project_name_3}/resource", auth_info_2
        )
        is False
    )
