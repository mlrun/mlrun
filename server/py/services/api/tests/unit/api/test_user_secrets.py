# Copyright 2025 Iguazio
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
from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun
import mlrun.common.schemas
import mlrun.common.types
import mlrun.errors

import services.api.api.endpoints.user_secrets as user_secrets

API_USER_SECRETS_PATH = "/user-secrets"
API_USER_SECRETS_TOKENS_PATH = API_USER_SECRETS_PATH + "/tokens"

_AUTH_USERNAME = "auth-user"
_AUTH_USER_ID = "auth-user-id"


@pytest.fixture
def auth_info() -> mlrun.common.schemas.AuthInfo:
    return mlrun.common.schemas.AuthInfo(username=_AUTH_USERNAME, user_id=_AUTH_USER_ID)


@pytest.fixture
def mock_query_global_resource_permissions(monkeypatch):
    """Fixture that returns a function to mock query_global_resource_permissions."""

    def _mock(
        expected_action: mlrun.common.schemas.AuthorizationAction,
        has_permission: bool,
    ):
        async def _fake_query_global_resource_permissions(
            self,
            resource_type,
            action,
            auth_info,
            raise_on_forbidden=True,
            resource_namespace=mlrun.common.schemas.AuthorizationResourceNamespace.resources,
        ) -> bool:
            assert auth_info.username == _AUTH_USERNAME
            assert action == expected_action
            assert (
                resource_type == mlrun.common.schemas.AuthorizationResourceTypes.tokens
            )
            assert (
                resource_namespace
                == mlrun.common.schemas.AuthorizationResourceNamespace.mgmt
            )
            return has_permission

        monkeypatch.setattr(
            user_secrets.framework.utils.auth.verifier.AuthVerifier,
            "query_global_resource_permissions",
            _fake_query_global_resource_permissions,
        )

    return _mock


def test_iguazio_v4_only_dependency(db: Session, client: TestClient):
    # Force unsupported auth mode
    mlrun.mlconf.httpdb.authentication.mode = (
        mlrun.common.types.AuthenticationMode.BASIC
    )

    # Pick an endpoint that includes the iguazio_v4_only dependency
    response = client.put(API_USER_SECRETS_TOKENS_PATH, json=[])

    assert response.status_code == HTTPStatus.BAD_REQUEST.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "has_system_admin_permission, username_param, expected_result, expected_error_message",
    [
        # Regular users
        # Username is None -> self
        (False, None, "auth-user", None),
        # Username is "" -> self
        (False, "", "auth-user", None),
        # Username is self -> self
        (False, "auth-user", "auth-user", None),
        # Username is self in different case -> self (Keycloak treats usernames case-insensitively)
        (False, "AUTH-USER", "AUTH-USER", None),
        (False, "Auth-User", "Auth-User", None),
        # Username is other user -> forbidden
        (
            False,
            "other-user",
            None,
            "Only system admins can list tokens for other users",
        ),
        # Username is wildcard -> forbidden
        (
            False,
            "*",
            None,
            "Only system admins can list tokens for all users",
        ),
        # Admin users
        # Username is None -> self
        (True, None, "auth-user", None),
        # Username is "" -> self
        (True, "", "auth-user", None),
        # Username is "*" -> all users (wildcard passed through)
        (
            True,
            "*",
            "*",
            None,
        ),
        # Username is some-user -> some-user
        (True, "some-user", "some-user", None),
        # Username is self -> self
        (True, "auth-user", "auth-user", None),
    ],
)
async def test_resolve_target_username_for_list(
    mock_query_global_resource_permissions,
    auth_info: mlrun.common.schemas.AuthInfo,
    has_system_admin_permission: bool,
    username_param: str | None,
    expected_result: str | None,
    expected_error_message: str | None,
):
    mock_query_global_resource_permissions(
        mlrun.common.schemas.AuthorizationAction.read, has_system_admin_permission
    )

    if expected_error_message:
        # When a non-admin provides `username`, we expect an error.
        with pytest.raises(
            mlrun.errors.MLRunAccessDeniedError,
            match=expected_error_message,
        ):
            await user_secrets._resolve_target_username_for_list_secret_tokens(
                auth_info, username_param
            )
    else:
        result = await user_secrets._resolve_target_username_for_list_secret_tokens(
            auth_info, username_param
        )
        assert result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "has_system_admin_permission, username_param, expected_result, expected_error_message",
    [
        # Regular users
        # Username is None -> self
        (False, None, "auth-user", None),
        # Username is "" -> self
        (False, "", "auth-user", None),
        # Username is self -> self
        (False, "auth-user", "auth-user", None),
        # Username is self in different case -> self (Keycloak treats usernames case-insensitively)
        (False, "AUTH-USER", "AUTH-USER", None),
        (False, "Auth-User", "Auth-User", None),
        # Username is other user -> forbidden
        (
            False,
            "other-user",
            None,
            "Only system admins can delete tokens for other users",
        ),
        # Admin users
        # Username is None -> self
        (True, None, "auth-user", None),
        # Username is "" -> self
        (True, "", "auth-user", None),
        # Username is specific -> that user
        (True, "some-user", "some-user", None),
        # Username is self -> self
        (True, "auth-user", "auth-user", None),
    ],
)
async def test_resolve_target_username_for_delete(
    mock_query_global_resource_permissions,
    auth_info: mlrun.common.schemas.AuthInfo,
    has_system_admin_permission: bool,
    username_param: str | None,
    expected_result: str | None,
    expected_error_message: str | None,
):
    mock_query_global_resource_permissions(
        mlrun.common.schemas.AuthorizationAction.delete, has_system_admin_permission
    )

    if expected_error_message:
        with pytest.raises(
            mlrun.errors.MLRunAccessDeniedError,
            match=expected_error_message,
        ):
            await user_secrets._resolve_target_username_for_delete_secret_tokens(
                auth_info, username_param
            )
    else:
        result = await user_secrets._resolve_target_username_for_delete_secret_tokens(
            auth_info, username_param
        )
        assert result == expected_result


@pytest.mark.asyncio
async def test_resolve_target_username_for_list_handles_none_auth_username(monkeypatch):
    # auth_info.username can be None (it's typed str | None); the self-check must not crash
    # and must deny non-admins who supply a concrete username.
    async def _no_admin(self, *args, **kwargs):
        return False

    monkeypatch.setattr(
        user_secrets.framework.utils.auth.verifier.AuthVerifier,
        "query_global_resource_permissions",
        _no_admin,
    )
    auth_info = mlrun.common.schemas.AuthInfo(username=None, user_id="x")

    with pytest.raises(
        mlrun.errors.MLRunAccessDeniedError,
        match="Only system admins can list tokens for other users",
    ):
        await user_secrets._resolve_target_username_for_list_secret_tokens(
            auth_info, "some-user"
        )


@pytest.mark.asyncio
async def test_resolve_target_username_for_delete_handles_none_auth_username(
    monkeypatch,
):
    async def _no_admin(self, *args, **kwargs):
        return False

    monkeypatch.setattr(
        user_secrets.framework.utils.auth.verifier.AuthVerifier,
        "query_global_resource_permissions",
        _no_admin,
    )
    auth_info = mlrun.common.schemas.AuthInfo(username=None, user_id="x")

    with pytest.raises(
        mlrun.errors.MLRunAccessDeniedError,
        match="Only system admins can delete tokens for other users",
    ):
        await user_secrets._resolve_target_username_for_delete_secret_tokens(
            auth_info, "some-user"
        )
