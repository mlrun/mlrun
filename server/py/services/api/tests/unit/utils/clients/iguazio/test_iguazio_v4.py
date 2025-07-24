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

import mlrun.errors
from tests.common_fixtures import aioresponses_mock

import framework.utils.clients.iguazio.v4
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
        kwargs["callback"] = callback
    if status_code:
        kwargs["status"] = status_code
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
    url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"
    patch_restful_request(
        aioresponses_mock,
        method=http.HTTPMethod.GET,
        url=url,
        status_code=http.HTTPStatus.UNAUTHORIZED.value,
    )
    with pytest.raises(mlrun.errors.MLRunUnauthorizedError) as exc:
        await maybe_coroutine(iguazio_client.verify_request_session(mock_request))

    assert exc.value.error_status_code == http.HTTPStatus.UNAUTHORIZED.value


@pytest.mark.parametrize("iguazio_client", ["async"], indirect=True)
@pytest.mark.parametrize(
    "headers",
    [
        {"cookie": "_oauth2_proxy=some-session-cookie"},  # cookie only
        # {"authorization": "Bearer some-jwt-token"},        # header only
        # {
        #     "cookie": "_oauth2_proxy=some-session-cookie",
        #     "authorization": "Bearer some-jwt-token"
        # },                                                # both present
    ],
)
@pytest.mark.asyncio
async def test_verify_request_session_success_ig4(
    api_url: str,
    iguazio_client,
    aioresponses_mock: aioresponses_mock,
    headers: dict,
):
    mock_request = fastapi.Request({"type": "http"})
    mock_request._headers = starlette.datastructures.Headers(headers)

    def _verify_session_with_body_mock(*args, **kwargs):
        # request_headers = kwargs["headers"]
        # for header_key, header_value in mock_request_headers.items():
        #     assert request_headers[header_key] == header_value
        return CallbackResult(payload=sample_user_info())

    patch_restful_request(
        aioresponses_mock,
        method=http.HTTPMethod.GET,
        url=api_url,
        callback=_verify_session_with_body_mock,
    )

    auth_info = await maybe_coroutine(
        iguazio_client.verify_request_session(mock_request)
    )

    assert auth_info.username == "dummy-user"
    assert auth_info.user_group_ids == ["dummy-group-id-g1", "dummy-group-id-g2"]


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
