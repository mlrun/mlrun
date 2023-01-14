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

import http.server
import typing
from unittest import mock

import aiohttp.client_exceptions
import aioresponses
import pytest
from aiohttp import BaseConnector, ClientConnectorError, ServerDisconnectedError

from mlrun.utils.async_http import AsyncClientWithRetry


@pytest.fixture
async def async_client():
    async with AsyncClientWithRetry() as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception",
    [
        ClientConnectorError(
            mock.MagicMock(
                code=500,
            ),
            ConnectionResetError(),
        ),
        ServerDisconnectedError(),
    ],
)
async def test_retry_os_exception_fail(async_client: AsyncClientWithRetry, exception):
    with aioresponses.aioresponses() as aiohttp_mock:
        max_retries = 3
        for i in range(max_retries):
            aiohttp_mock.get(
                "http://localhost:30678",
                exception=exception,
            )
        with pytest.raises(exception.__class__):
            async_client.retry_options.attempts = max_retries
            await async_client.get("http://localhost:30678")
        assert (
            len(list(aiohttp_mock.requests.values())[0]) == max_retries
        ), f"Expected {max_retries} retries"


@pytest.mark.asyncio
async def test_retry_os_exception_success(async_client: AsyncClientWithRetry):
    with aioresponses.aioresponses() as aiohttp_mock:
        max_retries = 3
        for i in range(max_retries - 1):
            aiohttp_mock.get(
                "http://localhost:30678",
                exception=ClientConnectorError(
                    mock.MagicMock(
                        code=500,
                    ),
                    ConnectionResetError(),
                ),
            )
            aiohttp_mock.get(
                "http://localhost:30678",
                status=200,
            )
        response = await async_client.get("http://localhost:30678")
        assert response.status == 200, "Expected to succeed after retries"
        assert (
            len(list(aiohttp_mock.requests.values())[0]) == max_retries - 1
        ), f"Expected {max_retries-1} retries"


@pytest.mark.asyncio
async def test_no_retry_on_blacklisted_method(async_client: AsyncClientWithRetry):
    with aioresponses.aioresponses() as aiohttp_mock:
        aiohttp_mock.post(
            "http://localhost:30678",
            status=500,
        )
        async_client.retry_options.blacklisted_methods = ["POST"]
        response = await async_client.post("http://localhost:30678")
        assert response.status == 500, "Expected to succeed after retries"

        # Do not retry because method is blacklisted
        aiohttp_mock.assert_called_once()


@pytest.mark.parametrize(
    "method, status_codes",
    [
        ("GET", [http.HTTPStatus.OK]),
        ("GET", [http.HTTPStatus.SERVICE_UNAVAILABLE, http.HTTPStatus.OK]),
        (
            "GET",
            [
                http.HTTPStatus.SERVICE_UNAVAILABLE,
                http.HTTPStatus.SERVICE_UNAVAILABLE,
                http.HTTPStatus.OK,
            ],
        ),
        (
            "GET",
            [
                http.HTTPStatus.SERVICE_UNAVAILABLE,
                http.HTTPStatus.SERVICE_UNAVAILABLE,
                http.HTTPStatus.SERVICE_UNAVAILABLE,
            ],
        ),
        ("POST", [http.HTTPStatus.OK]),
        # we don't retry on POST.
        # on top of that, we return the response even if it's an error, so asynchttp client don't raise an exception.
        ("POST", [http.HTTPStatus.SERVICE_UNAVAILABLE]),
    ],
)
@pytest.mark.asyncio
async def test_retry_method_status_codes(
    async_client: AsyncClientWithRetry,
    method: str,
    status_codes: typing.List[http.HTTPStatus],
):

    with aioresponses.aioresponses() as aiohttp_mock:
        for status_code in status_codes:
            aiohttp_mock.add("http://nothinghere", method=method, status=status_code)

        response = await async_client.request(method, "http://nothinghere")
        assert response.status == status_codes[-1], "response status is not as expected"

        # ensure we called the request the correct number of times
        assert len(list(aiohttp_mock.requests.values())[0]) == len(
            status_codes
        ), "Wrong number of retries"


@pytest.mark.asyncio
async def test_headers_serialization(async_client: AsyncClientWithRetry):
    """
    Header keys/values type must be str to be serializable
    This tests ensures we drop headers with 'None' values
    """
    BaseConnector.connect = mock.AsyncMock()
    with pytest.raises(aiohttp.client_exceptions.ClientOSError) as e:
        await async_client.get("http://nothinghere", headers={"x": None, "y": "z"})

    # assert the client tried to write the headers to the socket which means they were serialized
    assert "Cannot write to closing transport" in str(e.value)
