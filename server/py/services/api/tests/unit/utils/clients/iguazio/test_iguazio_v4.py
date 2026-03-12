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

import http
import unittest.mock
from contextlib import nullcontext as does_not_raise

import deepdiff
import httpx

# Skip the entire test module if running under Python < 3.11
import iguazio.schemas
import pytest
from aioresponses import CallbackResult

import mlrun.common.schemas
import mlrun.common.types
import mlrun.errors
from mlrun.utils.logger import context_id_var
from server.py.services.api.tests.unit.utils.clients.iguazio.conftest import (
    build_mock_request,
    patch_restful_request,
)
from tests.common_fixtures import aioresponses_mock

import framework.utils.clients.helpers as clients_helpers
import framework.utils.clients.iguazio.v4
from framework.utils.asyncio import maybe_coroutine

TEST_PROJECT_NAME = "test-project"
TEST_PROJECT_OWNER = "test-owner"
TEST_SERVICE_ACCOUNT_AUTH_HEADERS = {"Authorization": "Bearer test-sa-token"}


@pytest.fixture
def mock_service_account_auth_headers():
    """Mock the service account token client auth_headers property to avoid file access"""
    with unittest.mock.patch(
        "framework.utils.clients.service_account_token.Client.auth_headers",
        TEST_SERVICE_ACCOUNT_AUTH_HEADERS,
    ):
        yield TEST_SERVICE_ACCOUNT_AUTH_HEADERS


@pytest.mark.parametrize("iguazio_client", [("v4", "async")], indirect=True)
@pytest.mark.parametrize(
    "headers",
    [
        {},  # no cookie, no auth header
        {mlrun.common.schemas.HeaderNames.cookie: "some=thing"},  # wrong cookie
        {mlrun.common.schemas.HeaderNames.authorization: ""},  # empty header
    ],
)
@pytest.mark.asyncio
async def test_verify_request_session_failure(
    api_url: str,
    iguazio_client,
    aioresponses_mock: aioresponses_mock,
    headers: dict,
):
    mock_request = build_mock_request(headers)
    with pytest.raises(mlrun.errors.MLRunUnauthorizedError) as exc:
        await maybe_coroutine(iguazio_client.verify_request_session(mock_request))

    assert exc.value.error_status_code == http.HTTPStatus.UNAUTHORIZED.value, (
        "Expected 401 Unauthorized"
    )


@pytest.mark.parametrize("iguazio_client", [("v4", "async")], indirect=True)
@pytest.mark.parametrize(
    "headers",
    [
        {
            mlrun.common.schemas.HeaderNames.cookie: (
                f"{mlrun.common.schemas.CookieNames.oauth2_proxy}=some-session-cookie"
            )
        },  # cookie only
        {
            mlrun.common.schemas.HeaderNames.authorization: (
                f"{mlrun.common.schemas.AuthorizationHeaderPrefixes.bearer}some-jwt-token"
            )
        },  # header only
        {
            mlrun.common.schemas.HeaderNames.cookie: (
                f"{mlrun.common.schemas.CookieNames.oauth2_proxy}=some-session-cookie"
            ),
            mlrun.common.schemas.HeaderNames.authorization: (
                f"{mlrun.common.schemas.AuthorizationHeaderPrefixes.bearer}some-jwt-token"
            ),
        },  # both present
    ],
)
@pytest.mark.asyncio
async def test_verify_request_session_success(
    api_url: str,
    iguazio_client,
    aioresponses_mock: aioresponses_mock,
    headers: dict,
):
    mock_request = build_mock_request(headers)

    def _verify_session_with_body_mock(*args, **kwargs):
        response = sample_user_info()
        return CallbackResult(payload=response)

    url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"

    patch_restful_request(
        aioresponses_mock,
        method=mlrun.common.types.HTTPMethod.GET,
        url=url,
        callback=_verify_session_with_body_mock,
    )

    auth_info = await maybe_coroutine(
        iguazio_client.verify_request_session(mock_request)
    )

    assert auth_info.username == "dummy-user"
    assert auth_info.user_id == "dummy-user-id"
    assert auth_info.user_group_ids == ["dummy-group-id-g1", "dummy-group-id-g2"]


@pytest.mark.parametrize("iguazio_client", [("v4", "async")], indirect=True)
@pytest.mark.parametrize(
    "broken_response",
    [
        # Missing "username"
        {
            "metadata": {},
            "relationships": [
                {
                    "@type": "type.googleapis.com/usergroup.Group",
                    "metadata": {
                        "id": "dummy-group-id-g1",
                    },
                },
            ],
        },
        # Missing user ID
        {
            "metadata": {"username": "dummy-user"},
            "relationships": [
                {
                    "@type": "type.googleapis.com/usergroup.Group",
                    "metadata": {"id": "dummy-group-id-g1"},
                },
            ],
        },
        # metadata is not a dict
        {
            "metadata": "not-a-dict",
            "relationships": [
                {
                    "@type": "type.googleapis.com/usergroup.Group",
                    "metadata": {
                        "id": "dummy-group-id-g1",
                    },
                },
            ],
        },
        # relationships are not a list
        {
            "metadata": {"username": "dummy-user"},
            "relationships": "not-a-list",
        },
        {},  # Empty response
    ],
)
@pytest.mark.asyncio
async def test_verify_request_session_malformed_response(
    api_url: str,
    iguazio_client,
    aioresponses_mock: aioresponses_mock,
    broken_response: dict,
):
    """
    Covers both missing and malformed required fields in the session verification response.
    Fields:
    - 'metadata.username' must be a non-empty string
    - 'metadata' must be a dict
    - If 'relationships' exists, it must be a list (missing is OK)
    """
    headers = {
        mlrun.common.schemas.HeaderNames.cookie: (
            f"{mlrun.common.schemas.CookieNames.oauth2_proxy}=dummy-cookie"
        )
    }
    mock_request = build_mock_request(headers)

    def _mock_response(*args, **kwargs):
        return CallbackResult(payload=broken_response)

    url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"

    patch_restful_request(
        aioresponses_mock,
        method=mlrun.common.types.HTTPMethod.GET,
        url=url,
        callback=_mock_response,
    )

    with pytest.raises(mlrun.errors.MLRunUnauthorizedError) as exc:
        await maybe_coroutine(iguazio_client.verify_request_session(mock_request))

    assert exc.value.error_status_code == http.HTTPStatus.UNAUTHORIZED.value, (
        "Expected 401 Unauthorized"
    )


@pytest.mark.parametrize("iguazio_client", [("v4", "async")], indirect=True)
@pytest.mark.parametrize(
    "valid_response, expected_groups",
    [
        # Missing relationships → valid, no groups
        (
            {
                "metadata": {
                    "username": "dummy-user",
                    "id": "dummy-id",
                },
            },
            [],
        ),
        # Empty relationships list → valid, no groups
        (
            {
                "metadata": {
                    "username": "dummy-user",
                    "id": "dummy-id",
                },
                "relationships": [],
            },
            [],
        ),
    ],
)
@pytest.mark.asyncio
async def test_verify_request_session_valid_no_groups(
    api_url: str,
    iguazio_client,
    aioresponses_mock: aioresponses_mock,
    valid_response: dict,
    expected_groups: list[str],
):
    """
    Test valid responses where relationships are missing or empty.
    The user should be authenticated and group_ids should be an empty list.
    """
    headers = {
        mlrun.common.schemas.HeaderNames.cookie: (
            f"{mlrun.common.schemas.CookieNames.oauth2_proxy}=dummy-cookie"
        )
    }
    mock_request = build_mock_request(headers)

    def _mock_response(*args, **kwargs):
        return CallbackResult(payload=valid_response)

    url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"

    patch_restful_request(
        aioresponses_mock,
        method=mlrun.common.types.HTTPMethod.GET,
        url=url,
        callback=_mock_response,
    )

    auth_info = await maybe_coroutine(
        iguazio_client.verify_request_session(mock_request)
    )

    assert auth_info.username == "dummy-user"
    assert auth_info.user_id == "dummy-id"
    assert auth_info.user_group_ids == expected_groups


@pytest.mark.parametrize("iguazio_client", [("v4", "async")], indirect=True)
@pytest.mark.asyncio
async def test_verify_request_session_single_group_untyped(
    api_url: str,
    iguazio_client,
    aioresponses_mock: aioresponses_mock,
):
    headers = {
        mlrun.common.schemas.HeaderNames.cookie: (
            f"{mlrun.common.schemas.CookieNames.oauth2_proxy}=dummy-cookie"
        )
    }
    mock_request = build_mock_request(headers)

    # Include one valid group and one with invalid type
    response = {
        "metadata": {"username": "dummy-user", "id": "dummy-id"},
        "relationships": [
            {
                "@type": "type.googleapis.com/usergroup.Group",
                "metadata": {"id": "valid-group-id"},
            },
            {
                "@type": "some-other-type",
                "metadata": {"id": "ignored-id"},
            },
        ],
    }

    def _mock_response(*args, **kwargs):
        return CallbackResult(payload=response)

    url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"
    patch_restful_request(
        aioresponses_mock,
        method=mlrun.common.types.HTTPMethod.GET,
        url=url,
        callback=_mock_response,
    )

    auth_info = await maybe_coroutine(
        iguazio_client.verify_request_session(mock_request)
    )

    assert auth_info.username == "dummy-user"
    assert auth_info.user_id == "dummy-id"
    assert auth_info.user_group_ids == ["valid-group-id"]


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_delete_project_check_skips_igz_delete(
    iguazio_client, mock_service_account_auth_headers
) -> None:
    """Ensure IG4 check does not call igz delete project policies."""
    iguazio_client.delete_project(
        None,
        TEST_PROJECT_NAME,
        deletion_strategy=mlrun.common.schemas.DeletionStrategy.check,
    )
    iguazio_client._client.delete_project_policies.assert_not_called()


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
@pytest.mark.parametrize(
    "deletion_strategy",
    (
        mlrun.common.schemas.DeletionStrategy.restricted,
        mlrun.common.schemas.DeletionStrategy.cascading,
    ),
)
def test_delete_project_calls_igz_delete(
    iguazio_client,
    deletion_strategy: mlrun.common.schemas.DeletionStrategy,
    mock_service_account_auth_headers,
) -> None:
    """Ensure IG4 delete calls igz delete project policies once."""
    iguazio_client.delete_project(
        None, TEST_PROJECT_NAME, deletion_strategy=deletion_strategy
    )
    iguazio_client._client.delete_project_policies.assert_called_once_with(
        project=TEST_PROJECT_NAME
    )


def sample_user_info(username="dummy-user", user_id="dummy-user-id", group_ids=None):
    group_ids = group_ids or ["dummy-group-id-g1", "dummy-group-id-g2"]
    return {
        "metadata": {"resourceType": "user", "username": username, "id": user_id},
        "relationships": [
            {
                "@type": "type.googleapis.com/usergroup.Group",
                "metadata": {"id": gid},
            }
            for gid in group_ids
        ],
        "status": {"ctx": "dummy-ctx", "statusCode": http.HTTPStatus.OK.value},
    }


@pytest.mark.parametrize(
    "secret_token, expected_exception",
    [
        (None, mlrun.errors.MLRunInvalidArgumentError),  # None token
        (
            mlrun.common.schemas.SecretToken(name="t1", token=""),
            mlrun.errors.MLRunInvalidArgumentError,
        ),  # empty token
        (
            mlrun.common.schemas.SecretToken(name="t1", token="valid-token"),
            None,
        ),  # valid token
    ],
)
@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_refresh_access_token_cases(iguazio_client, secret_token, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            iguazio_client.refresh_access_token(secret_token)
    else:
        # simulate HTTP error
        iguazio_client._client.refresh_access_token.side_effect = httpx.HTTPStatusError(
            "Error",
            request=None,
            response=unittest.mock.MagicMock(
                status_code=401,
                json=lambda: {"status": {"errorMessage": "invalid", "ctx": "dummy"}},
            ),
        )
        with pytest.raises(mlrun.errors.MLRunUnauthorizedError):
            iguazio_client.refresh_access_token(secret_token)


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_refresh_access_token_success(
    iguazio_client, mock_service_account_auth_headers
):
    secret_token = mlrun.common.schemas.SecretToken(
        name="test-token", token="valid-token"
    )

    # Should not raise
    iguazio_client.refresh_access_token(secret_token)

    iguazio_client._client.refresh_access_token.assert_called_once()
    called_options = iguazio_client._client.refresh_access_token.call_args[1]["options"]
    assert called_options.refresh_token == "valid-token"


@pytest.mark.parametrize(
    "secret_tokens, expected_exception",
    [
        ([], mlrun.errors.MLRunInvalidArgumentError),  # empty list
        (
            [mlrun.common.schemas.SecretToken(name="t1", token="")],
            mlrun.errors.MLRunInvalidArgumentError,
        ),  # empty token in list
        (
            [
                mlrun.common.schemas.SecretToken(name="t1", token="token1"),
                mlrun.common.schemas.SecretToken(name="t2", token="token2"),
            ],
            None,
        ),  # valid tokens
    ],
)
@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_refresh_access_tokens_cases(iguazio_client, secret_tokens, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            iguazio_client.refresh_access_tokens(secret_tokens)
    else:
        # simulate HTTP error
        iguazio_client._client.refresh_access_tokens.side_effect = (
            httpx.HTTPStatusError(
                "Error",
                request=None,
                response=unittest.mock.MagicMock(
                    status_code=401,
                    json=lambda: {
                        "status": {"errorMessage": "invalid", "ctx": "dummy"}
                    },
                ),
            )
        )
        with pytest.raises(mlrun.errors.MLRunUnauthorizedError):
            iguazio_client.refresh_access_tokens(secret_tokens)


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_revoke_offline_token_success(
    iguazio_client, mock_service_account_auth_headers
):
    token = "valid-token"
    request_headers = {
        mlrun.common.schemas.HeaderNames.authorization: f"{mlrun.common.schemas.AuthorizationHeaderPrefixes.bearer}123",
    }

    iguazio_client.revoke_offline_token(token, request_headers)

    iguazio_client._client.revoke_offline_token.assert_called_once()


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_resolve_token_from_igz_yml_success(iguazio_client, monkeypatch):
    """Test successful token resolution from igz.yml content."""
    igz_yml_content = "secretTokens:\n- name: my-token\n  token: jwt-value\n"

    # Mock iguazio.Client for the token file client
    mock_token_client = unittest.mock.Mock()
    mock_token_client.get_refresh_token.return_value = ("my-token", "jwt-value")

    with unittest.mock.patch("iguazio.Client", return_value=mock_token_client):
        result = iguazio_client.resolve_token_from_igz_yml(
            igz_yml_content, "test-user", "my-token"
        )

    assert result == "my-token"
    mock_token_client.get_refresh_token.assert_called_once()


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_resolve_token_from_igz_yml_auto_discovery(iguazio_client, monkeypatch):
    """Test auto-discovery mode returns first valid token."""
    igz_yml_content = "secretTokens:\n- name: default\n  token: jwt-default\n- name: other\n  token: jwt-other\n"

    mock_token_client = unittest.mock.Mock()
    mock_token_client.get_refresh_token.return_value = ("default", "jwt-default")

    with unittest.mock.patch(
        "iguazio.Client", return_value=mock_token_client
    ) as mock_class:
        result = iguazio_client.resolve_token_from_igz_yml(
            igz_yml_content, "test-user", None
        )

    assert result == "default"
    # Verify token_name=None for auto-discovery
    call_kwargs = mock_class.call_args.kwargs
    assert call_kwargs["token_name"] is None


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_resolve_token_from_igz_yml_token_not_found(iguazio_client):
    """Test MLRunNotFoundError when specific token is not found."""
    igz_yml_content = "secretTokens:\n- name: other-token\n  token: jwt-value\n"

    with unittest.mock.patch(
        "iguazio.Client", side_effect=ValueError("Token 'my-token' not found")
    ):
        with pytest.raises(
            mlrun.errors.MLRunNotFoundError, match="not found or invalid"
        ):
            iguazio_client.resolve_token_from_igz_yml(
                igz_yml_content, "test-user", "my-token"
            )


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_resolve_token_from_igz_yml_no_valid_tokens(iguazio_client):
    """Test MLRunNotFoundError when no valid tokens are found in auto-discovery."""
    igz_yml_content = "secretTokens:\n- name: expired\n  token: expired-jwt\n"

    with unittest.mock.patch(
        "iguazio.Client", side_effect=RuntimeError("No valid tokens found")
    ):
        with pytest.raises(
            mlrun.errors.MLRunNotFoundError, match="No valid tokens found"
        ):
            iguazio_client.resolve_token_from_igz_yml(
                igz_yml_content, "test-user", None
            )


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_resolve_token_from_igz_yml_sdk_returns_none(iguazio_client):
    """Test MLRunNotFoundError when SDK returns None."""
    igz_yml_content = "secretTokens:\n- name: some-token\n  token: jwt-value\n"

    mock_token_client = unittest.mock.Mock()
    mock_token_client.get_refresh_token.return_value = (None, None)

    with unittest.mock.patch("iguazio.Client", return_value=mock_token_client):
        with pytest.raises(
            mlrun.errors.MLRunNotFoundError, match="No valid tokens found"
        ):
            iguazio_client.resolve_token_from_igz_yml(
                igz_yml_content, "test-user", None
            )


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_create_project(
    mock_session, iguazio_client, igv4_auth_info, mock_service_account_auth_headers
):
    project = _generate_igv4_project()

    iguazio_client.create_project(mock_session, project, auth_info=igv4_auth_info)
    iguazio_client._client.create_default_project_policies.assert_called_once_with(
        project=TEST_PROJECT_NAME
    )


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
@pytest.mark.parametrize("patch_mode", mlrun.common.schemas.PatchMode)
@pytest.mark.parametrize("owner", [TEST_PROJECT_OWNER, None])
def test_patch_project(
    owner,
    patch_mode,
    mock_session,
    iguazio_client,
    igv4_auth_info,
    mock_service_account_auth_headers,
):
    project = _generate_igv4_project(owner=owner)

    iguazio_client.patch_project(
        mock_session,
        TEST_PROJECT_NAME,
        project.dict(),
        patch_mode,
        auth_info=igv4_auth_info,
    )

    if not owner:
        iguazio_client._client.update_project_owner.assert_not_called()
    else:
        iguazio_client._client.update_project_owner.assert_called_once_with(
            project=TEST_PROJECT_NAME,
            options=iguazio.schemas.UpdateProjectOwnerOptionsV1(owner=owner),
        )


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_patch_project_forbidden_raises_access_denied(
    mock_session,
    iguazio_client,
    igv4_auth_info,
    mock_service_account_auth_headers,
):
    """Reproducer: patching project owner with insufficient permissions must raise
    MLRunAccessDeniedError (not MLRunInternalServerError)."""
    iguazio_client._client.update_project_owner.side_effect = (
        _generate_igv4_httpx_exception(
            "Not allowed to update resource",
            httpx.codes.FORBIDDEN,
        )
    )
    project = _generate_igv4_project()

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        iguazio_client.patch_project(
            mock_session,
            TEST_PROJECT_NAME,
            project.dict(),
            auth_info=igv4_auth_info,
        )


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
@pytest.mark.parametrize("owner", [TEST_PROJECT_OWNER, None])
@pytest.mark.parametrize("project_exists", [True, False])
def test_store_project(
    project_exists,
    owner,
    mock_session,
    iguazio_client,
    igv4_auth_info,
    mock_service_account_auth_headers,
):
    project = _generate_igv4_project(owner=owner)

    if project_exists:
        # Simulate 409 Conflict when trying to create project policies that already exist
        iguazio_client._client.create_default_project_policies.side_effect = (
            _generate_igv4_httpx_exception(
                "Project policies already exist",
                httpx.codes.CONFLICT,
            )
        )

    iguazio_client.store_project(
        mock_session, TEST_PROJECT_NAME, project, auth_info=igv4_auth_info
    )

    # Policies creation is always attempted
    iguazio_client._client.create_default_project_policies.assert_called_once_with(
        project=TEST_PROJECT_NAME
    )

    # store_project should not update the owner.
    # Owner updates should only happen via explicit patch_project calls.
    iguazio_client._client.update_project_owner.assert_not_called()


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_delete_project(mock_session, iguazio_client, igv4_auth_info):
    auth_headers = {"test": "test"}
    with unittest.mock.patch(
        "framework.utils.clients.service_account_token.Client.auth_headers",
        auth_headers,
    ):
        iguazio_client.delete_project(
            mock_session, TEST_PROJECT_NAME, auth_info=igv4_auth_info
        )

    iguazio_client._client.delete_project_policies.assert_called_once_with(
        project=TEST_PROJECT_NAME
    )


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_try_callback_with_httpx_exceptions_generic_exception(
    iguazio_client, mock_service_account_auth_headers
):
    """Generic (non-HTTP) exceptions use the fallback exception_type."""
    fallback_exception_type = mlrun.errors.MLRunInternalServerError
    failure_message = "Final failure message"

    def callback():
        raise Exception()

    with pytest.raises(fallback_exception_type, match=failure_message):
        iguazio_client._try_callback_with_httpx_exceptions(
            callback, fallback_exception_type, failure_message
        )


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
@pytest.mark.parametrize(
    "http_status_code, expected_exception_type",
    [
        (httpx.codes.BAD_REQUEST, mlrun.errors.MLRunBadRequestError),
        (httpx.codes.UNAUTHORIZED, mlrun.errors.MLRunUnauthorizedError),
        (httpx.codes.FORBIDDEN, mlrun.errors.MLRunAccessDeniedError),
        (httpx.codes.NOT_FOUND, mlrun.errors.MLRunNotFoundError),
        (httpx.codes.CONFLICT, mlrun.errors.MLRunConflictError),
        (
            httpx.codes.INTERNAL_SERVER_ERROR,
            mlrun.errors.MLRunInternalServerError,
        ),
    ],
)
def test_try_callback_with_httpx_exceptions_maps_status_code(
    iguazio_client,
    http_status_code,
    expected_exception_type,
    mock_service_account_auth_headers,
):
    """HTTP errors are mapped to the correct MLRun error type based on status code,
    regardless of the fallback exception_type."""
    failure_message = "Operation failed"
    error_message = "Upstream error"
    ctx = "test-ctx"

    def callback():
        raise _generate_igv4_httpx_exception(error_message, http_status_code, ctx)

    with pytest.raises(expected_exception_type) as exc_info:
        iguazio_client._try_callback_with_httpx_exceptions(
            callback,
            mlrun.errors.MLRunInternalServerError,
            failure_message,
        )
    assert failure_message in str(exc_info.value)
    assert error_message in str(exc_info.value)


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_try_callback_with_httpx_exceptions_unmapped_status_falls_back_to_exception_type(
    iguazio_client, mock_service_account_auth_headers
):
    """HTTP status codes not in STATUS_ERRORS fall back to the caller-supplied exception_type."""
    failure_message = "Operation failed"
    error_message = "Too many requests"
    ctx = "test-ctx"
    fallback_type = mlrun.errors.MLRunUnauthorizedError

    def callback():
        raise _generate_igv4_httpx_exception(error_message, 429, ctx)

    with pytest.raises(fallback_type) as exc_info:
        iguazio_client._try_callback_with_httpx_exceptions(
            callback, fallback_type, failure_message
        )
    assert failure_message in str(exc_info.value)
    assert error_message in str(exc_info.value)


@pytest.mark.parametrize(
    "existing_headers, expected_headers",
    [
        (
            # Not overriding existing headers
            {"Authorization": "Bearer token123"},
            {
                "Authorization": "Bearer token123",
                mlrun.common.schemas.HeaderNames.igz_ctx: "test-context-id-v4-12345",
            },
        ),
        (
            # Enriching empty headers
            {},
            {mlrun.common.schemas.HeaderNames.igz_ctx: "test-context-id-v4-12345"},
        ),
        (
            # Overriding existing context ID header
            {
                mlrun.common.schemas.HeaderNames.igz_ctx: "existing-context-id",
            },
            {
                mlrun.common.schemas.HeaderNames.igz_ctx: "existing-context-id",
            },
        ),
    ],
)
def test_enrich_headers_injects_context_id(existing_headers, expected_headers):
    """Verify that enrich_headers injects context ID into headers"""

    context_id = "test-context-id-v4-12345"
    headers = existing_headers

    token = context_id_var.set(context_id)
    try:
        enriched_headers = clients_helpers.enrich_headers(headers)
        # enrich_headers modifies in place, so check the original dict
        assert deepdiff.DeepDiff(enriched_headers, expected_headers) == {}
    finally:
        context_id_var.reset(token)


def test_no_context_id_when_not_set():
    """Verify that no context ID is added when context_id_var is not set"""
    headers = {"Authorization": "Bearer token123"}

    token = context_id_var.set(None)
    try:
        clients_helpers.enrich_headers(headers)
        # Should not add context ID header
        assert mlrun.common.schemas.HeaderNames.igz_ctx not in headers
    finally:
        context_id_var.reset(token)


@pytest.mark.parametrize("iguazio_client", [("v4", "sync")], indirect=True)
def test_context_id_passed_to_with_headers(
    mock_session, iguazio_client, igv4_auth_info, mock_service_account_auth_headers
):
    """Verify that context ID is passed through with_headers context manager"""
    context_id = "v4-callback-context-id"
    project = _generate_igv4_project()

    token = context_id_var.set(context_id)
    try:
        iguazio_client.create_project(mock_session, project, auth_info=igv4_auth_info)

        # Verify with_headers was called
        iguazio_client._client.with_headers.assert_called()

        # Get the headers that were passed to with_headers
        call_args = iguazio_client._client.with_headers.call_args
        if call_args and call_args[0]:
            headers_arg = call_args[0][0]
            # The headers should contain the context ID
            assert mlrun.common.schemas.HeaderNames.igz_ctx in headers_arg
            assert headers_arg[mlrun.common.schemas.HeaderNames.igz_ctx] == context_id
    finally:
        context_id_var.reset(token)


@pytest.fixture
def igv4_auth_info() -> mlrun.common.schemas.AuthInfo:
    request_headers = {
        mlrun.common.schemas.HeaderNames.authorization: f"{mlrun.common.schemas.AuthorizationHeaderPrefixes.bearer}123",
    }
    yield mlrun.common.schemas.AuthInfo(request_headers=request_headers)


@pytest.fixture
def mock_session() -> unittest.mock.MagicMock:
    yield unittest.mock.MagicMock()


def _generate_igv4_project(
    name: str = TEST_PROJECT_NAME,
    owner: str = TEST_PROJECT_OWNER,
) -> mlrun.common.schemas.Project:
    return mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=name),
        spec=mlrun.common.schemas.ProjectSpec(owner=owner),
    )


def _generate_igv4_httpx_exception(
    error_message: str,
    status_code: int,
    ctx: str = "test-ctx",
) -> httpx.HTTPStatusError:
    mock_response = unittest.mock.MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {
        "status": {
            "ctx": ctx,
            "statusCode": status_code,
            "errorMessage": error_message,
        }
    }

    return httpx.HTTPStatusError(
        message=error_message,
        request=unittest.mock.MagicMock(),
        response=mock_response,
    )


@pytest.mark.parametrize(
    "response_body, expected_result, expectation",
    [
        # Valid response
        (
            {
                "metadata": {
                    "username": "test-user",
                    "id": "test-id",
                    "resourceType": "user",
                },
                "relationships": [
                    {
                        "@type": "type.googleapis.com/usergroup.Group",
                        "metadata": {"id": "group1"},
                    },
                    {
                        "@type": "type.googleapis.com/usergroup.Group",
                        "metadata": {"id": "group2"},
                    },
                ],
            },
            ("test-user", "test-id", ["group1", "group2"], "user"),
            does_not_raise(),
        ),
        # Missing username
        (
            {
                "metadata": {"id": "test-id"},
                "relationships": [],
            },
            None,
            pytest.raises(mlrun.errors.MLRunUnauthorizedError),
        ),
        # Missing user ID
        (
            {
                "metadata": {"username": "test-user"},
                "relationships": [],
            },
            None,
            pytest.raises(mlrun.errors.MLRunUnauthorizedError),
        ),
        # Invalid relationships format
        (
            {
                "metadata": {"username": "test-user", "id": "test-id"},
                "relationships": "invalid-format",
            },
            None,
            pytest.raises(mlrun.errors.MLRunUnauthorizedError),
        ),
        # No relationships (valid case)
        (
            {
                "metadata": {"username": "test-user", "id": "test-id"},
            },
            ("test-user", "test-id", [], "user"),
            does_not_raise(),
        ),
        # Relationships with invalid group type
        (
            {
                "metadata": {"username": "test-user", "id": "test-id"},
                "relationships": [
                    {
                        "@type": "invalid-type",
                        "metadata": {"id": "ignored-group"},
                    }
                ],
            },
            ("test-user", "test-id", [], "user"),
            does_not_raise(),
        ),
    ],
)
def test_parse_auth_response_data(response_body, expected_result, expectation):
    with expectation:
        result = framework.utils.clients.iguazio.v4.Client._parse_auth_response_data(
            response_body
        )
        assert result == expected_result
