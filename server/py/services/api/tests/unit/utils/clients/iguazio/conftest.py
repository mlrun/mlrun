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
import typing

import fastapi
import starlette.datastructures

from tests.common_fixtures import aioresponses_mock


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
        kwargs["status"] = http.HTTPStatus.OK.value
    aioresponses_mock.add(
        url,
        method,
        **kwargs,
    )


def build_mock_request(headers: dict) -> fastapi.Request:
    request = fastapi.Request({"type": "http"})
    request._headers = starlette.datastructures.Headers(headers)
    return request
