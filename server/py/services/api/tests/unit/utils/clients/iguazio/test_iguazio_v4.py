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

import http
import typing

import fastapi
import pytest
import starlette.datastructures
from aioresponses import CallbackResult

import mlrun.common.schemas
import mlrun.errors
from tests.common_fixtures import aioresponses_mock

from framework.utils.asyncio import maybe_coroutine


def patch_restful_request(
    aioresponses_mock: aioresponses_mock,
    method: str,
    url: str,
    callback: typing.Optional[typing.Callable] = None,
    status_code: typing.Optional[int] = None,
):
    """
    Consolidating the requests_mock / aioresponses library to mock a RESTful request.
    """
    kwargs = {}
    if callback:
        # The callback should produce CallbackResult with status set explicitly
        kwargs["callback"] = callback
    elif status_code:
        # If no callback, set status and empty body
        kwargs["status"] = status_code
    else:
        # Default 200 OK with empty body
        kwargs["status"] = 200
    aioresponses_mock.add(
        url,
        method,
        **kwargs,
    )


@pytest.mark.parametrize("iguazio_client", [("v4", "async")], indirect=True)
@pytest.mark.parametrize(
    "headers",
    [
        {},  # no cookie, no auth header
        {"cookie": "some=thing"},  # wrong cookie
        {"authorization": ""},  # empty header
    ],
)
@pytest.mark.asyncio
async def test_verify_request_session_failure(
    api_url: str,
    iguazio_client,
    aioresponses_mock: aioresponses_mock,
    headers: dict,
):
    mock_request = fastapi.Request({"type": "http"})
    mock_request._headers = starlette.datastructures.Headers(headers)
    with pytest.raises(mlrun.errors.MLRunUnauthorizedError) as exc:
        await maybe_coroutine(iguazio_client.verify_request_session(mock_request))

    assert exc.value.error_status_code == http.HTTPStatus.UNAUTHORIZED.value


@pytest.mark.parametrize("iguazio_client", [("v4", "async")], indirect=True)
@pytest.mark.parametrize(
    "headers",
    [
        {
            "cookie": f"{mlrun.common.schemas.CookieNames.oauth2_proxy}=some-session-cookie"
        },  # cookie only
        {
            mlrun.common.schemas.HeaderNames.authorization: (
                f"{mlrun.common.schemas.AuthorizationHeaderPrefixes.bearer}some-jwt-token"
            )
        },  # header only
        {
            "cookie": f"{mlrun.common.schemas.CookieNames.oauth2_proxy}=some-session-cookie",
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
    mock_request = fastapi.Request({"type": "http"})
    mock_request._headers = starlette.datastructures.Headers(headers)

    def _verify_session_with_body_mock(*args, **kwargs):
        response = sample_user_info()
        return CallbackResult(payload=response)

    url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"

    patch_restful_request(
        aioresponses_mock,
        method=http.HTTPMethod.GET,
        url=url,
        callback=_verify_session_with_body_mock,
    )

    auth_info = await maybe_coroutine(
        iguazio_client.verify_request_session(mock_request)
    )

    assert auth_info.username == "dummy-user"
    assert auth_info.user_group_ids == ["dummy-group-id-g1", "dummy-group-id-g2"]


@pytest.mark.parametrize("iguazio_client", [("v4", "async")], indirect=True)
@pytest.mark.parametrize("missing_field", ["username", "groups"])
@pytest.mark.asyncio
async def test_verify_request_session_failure_missing_username(
    api_url: str,
    iguazio_client,
    aioresponses_mock: aioresponses_mock,
    missing_field: str,
):
    """
    Test case where the response is missing the 'username' field — should raise error.
    """
    mock_request = fastapi.Request({"type": "http"})
    mock_request._headers = starlette.datastructures.Headers(
        {"cookie": f"{mlrun.common.schemas.CookieNames.oauth2_proxy}=dummy-cookie"}
    )

    def _verify_session_mock_missing_user_info(*args, **kwargs):
        response = sample_user_info()
        if missing_field == "username":
            del response["metadata"]["username"]
        elif missing_field == "groups":
            del response["relationships"]
        return CallbackResult(payload=response)

    url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"

    patch_restful_request(
        aioresponses_mock,
        method=http.HTTPMethod.GET,
        url=url,
        callback=_verify_session_mock_missing_user_info,
    )

    with pytest.raises(mlrun.errors.MLRunUnauthorizedError) as exc:
        await maybe_coroutine(iguazio_client.verify_request_session(mock_request))

    assert exc.value.error_status_code == http.HTTPStatus.UNAUTHORIZED.value


# @pytest.mark.parametrize("iguazio_client", [("v4", "async")], indirect=True)
# @pytest.mark.asyncio
# async def test_verify_request_session_handle_error_response(
#     api_url: str,
#     iguazio_client,
#     aioresponses_mock: aioresponses_mock,
# ):
#     mock_request = fastapi.Request({"type": "http"})
#     mock_request._headers = starlette.datastructures.Headers(
#         {"cookie": f"{mlrun.common.schemas.CookieNames.oauth2_proxy}=dummy-cookie"}
#     )
#
#     def _verify_session_mock(*args, **kwargs):
#         response = sample_user_info()
#         return CallbackResult(
#             payload=response,
#             status=http.HTTPStatus.UNAUTHORIZED.value,
#         )
#
#     url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"
#
#     patch_restful_request(
#         aioresponses_mock,
#         method=http.HTTPMethod.GET,
#         url=url,
#         callback=_verify_session_mock,
#     )
#
#     with pytest.raises(mlrun.errors.MLRunUnauthorizedError) as exc:
#         await maybe_coroutine(iguazio_client.verify_request_session(mock_request))
#
#     assert exc.value.error_status_code == http.HTTPStatus.UNAUTHORIZED.value
#
#
def sample_user_info():
    return {
        "metadata": {"resourceType": "user", "username": "dummy-user"},
        "relationships": [
            {
                "@type": "type.googleapis.com/group.Group",
                "metadata": {
                    "id": "dummy-group-id-g1",
                },
            },
            {
                "@type": "type.googleapis.com/group.Group",
                "metadata": {
                    "id": "dummy-group-id-g2",
                },
            },
        ],
        "status": {"ctx": "dummy-ctx", "statusCode": http.HTTPStatus.OK.value},
    }
