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
import sys
import unittest.mock

import httpx
import pytest
from aioresponses import CallbackResult

# Skip the entire test module if running under Python < 3.11
if sys.version_info < (3, 11):
    pytest.skip("Requires Python 3.11+", allow_module_level=True)

import mlrun.common.schemas
import mlrun.common.types
import mlrun.errors
from server.py.services.api.tests.unit.utils.clients.iguazio.conftest import (
    build_mock_request,
    patch_restful_request,
)
from tests.common_fixtures import aioresponses_mock

from framework.utils.asyncio import maybe_coroutine


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

    assert (
        exc.value.error_status_code == http.HTTPStatus.UNAUTHORIZED.value
    ), "Expected 401 Unauthorized"


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
                    "@type": "type.googleapis.com/group.Group",
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
                    "@type": "type.googleapis.com/group.Group",
                    "metadata": {"id": "dummy-group-id-g1"},
                },
            ],
        },
        # metadata is not a dict
        {
            "metadata": "not-a-dict",
            "relationships": [
                {
                    "@type": "type.googleapis.com/group.Group",
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

    assert (
        exc.value.error_status_code == http.HTTPStatus.UNAUTHORIZED.value
    ), "Expected 401 Unauthorized"


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
                "@type": "type.googleapis.com/group.Group",
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


def sample_user_info(username="dummy-user", user_id="dummy-user-id", group_ids=None):
    group_ids = group_ids or ["dummy-group-id-g1", "dummy-group-id-g2"]
    return {
        "metadata": {"resourceType": "user", "username": username, "id": user_id},
        "relationships": [
            {
                "@type": "type.googleapis.com/group.Group",
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
def test_refresh_access_token_success(iguazio_client):
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
def test_revoke_offline_token_success(iguazio_client):
    token = "valid-token"
    request_headers = {
        mlrun.common.schemas.HeaderNames.authorization: f"{mlrun.common.schemas.AuthorizationHeaderPrefixes.bearer}123",
    }

    iguazio_client.revoke_offline_token(token, request_headers)

    iguazio_client._client.set_override_auth_headers.assert_called_once_with(
        request_headers
    )
    iguazio_client._client.revoke_offline_token.assert_called_once()
