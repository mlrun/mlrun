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
import unittest.mock
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


def _auth_info() -> mlrun.common.schemas.AuthInfo:
    return mlrun.common.schemas.AuthInfo(username=_AUTH_USERNAME, user_id=_AUTH_USER_ID)


def _mock_is_system_admin(
    monkeypatch,
    expected_action: mlrun.common.schemas.AuthorizationAction,
    is_admin: bool,
):
    async def _fake_is_system_admin(
        auth_info: mlrun.common.schemas.AuthInfo,
        action_to_check: mlrun.common.schemas.AuthorizationAction,
    ) -> bool:
        assert auth_info.username == _AUTH_USERNAME
        assert action_to_check == expected_action
        return is_admin

    monkeypatch.setattr(user_secrets, "_is_system_admin", _fake_is_system_admin)


def test_iguazio_v4_only_dependency(db: Session, client: TestClient):
    # Force unsupported auth mode
    orig_mode = mlrun.mlconf.httpdb.authentication.mode
    mlrun.mlconf.httpdb.authentication.mode = (
        mlrun.common.types.AuthenticationMode.BASIC
    )

    # Pick an endpoint that includes the iguazio_v4_only dependency
    response = client.put(API_USER_SECRETS_TOKENS_PATH, json=[])

    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    mlrun.mlconf.httpdb.authentication.mode = orig_mode


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action, allowed",
    [
        (mlrun.common.schemas.AuthorizationAction.read, True),  # System admin
        (mlrun.common.schemas.AuthorizationAction.read, False),  # Non-system admin
        (mlrun.common.schemas.AuthorizationAction.delete, True),  # System admin
        (mlrun.common.schemas.AuthorizationAction.delete, False),  # Non-system admin
    ],
)
async def test_is_system_admin_check(
    monkeypatch,
    action: mlrun.common.schemas.AuthorizationAction,
    allowed: bool,
):
    # "System admin" for user-secrets token operations is defined as:
    # having permission on the mgmt-scoped "tokens" resource.
    query_mock = unittest.mock.AsyncMock(return_value=allowed)
    monkeypatch.setattr(
        user_secrets.framework.utils.auth.verifier.AuthVerifier(),
        "query_resource_permissions",
        query_mock,
    )

    auth_info = _auth_info()

    result = await user_secrets._is_system_admin(auth_info, action)

    assert result is allowed
    query_mock.assert_called_once_with(
        mlrun.common.schemas.AuthorizationResourceTypes.tokens,
        "",
        action,
        auth_info,
        raise_on_forbidden=False,
        resource_namespace=mlrun.common.schemas.AuthorizationResourceNamespace.mgmt,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_admin, username_param, expected_result, expected_error_message",
    [
        (
            False,
            None,
            "auth-user",
            None,
        ),  # regular user, list without username -> self
        (
            True,
            None,
            None,
            None,
        ),  # admin, list without username -> all users
        (
            True,
            "some-user",
            "some-user",
            None,
        ),  # admin, list with username -> some-user
        (
            False,
            "some-user",
            None,
            "Only system admins can read tokens for other users",
        ),  # regular user, list with username -> forbidden
        (
            False,
            "auth-user",
            None,
            "Only system admins can read tokens for other users",
        ),  # regular user, list with username=self -> forbidden (username param disallowed)
    ],
)
async def test_resolve_target_username_for_list(
    monkeypatch,
    is_admin: bool,
    username_param: str | None,
    expected_result: str | None,
    expected_error_message: str | None,
):
    _mock_is_system_admin(
        monkeypatch, mlrun.common.schemas.AuthorizationAction.read, is_admin
    )
    auth_info = _auth_info()

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
    "is_admin, username_param, expected_result, expected_error_message",
    [
        (
            False,
            None,
            "auth-user",
            None,
        ),  # regular user, revoke without username -> self
        (
            True,
            None,
            "auth-user",
            None,
        ),  # admin, revoke without username -> self
        (
            True,
            "some-user",
            "some-user",
            None,
        ),  # admin, revoke with username -> some-user
        (
            False,
            "some-user",
            None,
            "Only system admins can delete tokens for other users",
        ),  # regular user, revoke with username -> forbidden
        (
            False,
            "auth-user",
            None,
            "Only system admins can delete tokens for other users",
        ),  # regular user, revoke with username=self -> forbidden (username param disallowed)
    ],
)
async def test_resolve_target_username_for_revoke(
    monkeypatch,
    is_admin: bool,
    username_param: str | None,
    expected_result: str | None,
    expected_error_message: str | None,
):
    _mock_is_system_admin(
        monkeypatch, mlrun.common.schemas.AuthorizationAction.delete, is_admin
    )
    auth_info = _auth_info()

    if expected_error_message:
        with pytest.raises(
            mlrun.errors.MLRunAccessDeniedError,
            match=expected_error_message,
        ):
            await user_secrets._resolve_target_username_for_revoke_secret_tokens(
                auth_info, username_param
            )
    else:
        result = await user_secrets._resolve_target_username_for_revoke_secret_tokens(
            auth_info, username_param
        )
        assert result == expected_result
