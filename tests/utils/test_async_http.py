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

import http.server
from contextlib import nullcontext as does_not_raise
from unittest import mock

import aioresponses
import pytest
import pytest_asyncio
from aiohttp import ClientConnectorError, ServerDisconnectedError

from mlrun.utils.async_http import AsyncClientWithRetry
from tests.common_fixtures import aioresponses_mock


@pytest_asyncio.fixture
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
async def test_retry_os_exception_fail(
    async_client: AsyncClientWithRetry, aioresponses_mock: aioresponses_mock, exception
):
    max_retries = 3
    for i in range(max_retries):
        aioresponses_mock.get(
            "http://localhost:30678",
            exception=exception,
        )
    with pytest.raises(exception.__class__):
        async_client.retry_options.attempts = max_retries
        await async_client.get("http://localhost:30678")
    assert (
        aioresponses_mock.called_times() == max_retries
    ), f"Expected {max_retries} retries"


@pytest.mark.asyncio
async def test_retry_os_exception_success(
    async_client: AsyncClientWithRetry, aioresponses_mock: aioresponses_mock
):
    max_retries = 3
    for i in range(max_retries - 1):
        aioresponses_mock.get(
            "http://localhost:30678",
            exception=ClientConnectorError(
                mock.MagicMock(
                    code=500,
                ),
                ConnectionResetError(),
            ),
        )
        aioresponses_mock.get(
            "http://localhost:30678",
            status=200,
        )
    response = await async_client.get("http://localhost:30678")
    assert response.status == 200, "Expected to succeed after retries"
    assert (
        aioresponses_mock.called_times() == max_retries - 1
    ), f"Expected {max_retries - 1} retries"


@pytest.mark.asyncio
async def test_no_retry_on_blacklisted_method(
    async_client: AsyncClientWithRetry, aioresponses_mock: aioresponses_mock
):
    aioresponses_mock.post(
        "http://localhost:30678",
        status=500,
    )
    async_client.retry_options.blacklisted_methods = ["POST"]
    response = await async_client.post("http://localhost:30678")
    assert response.status == 500, "Expected to succeed after retries"

    # Do not retry because method is blacklisted
    aioresponses_mock.assert_called_once()


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
    aioresponses_mock: aioresponses_mock,
    method: str,
    status_codes: list[http.HTTPStatus],
):
    for status_code in status_codes:
        aioresponses_mock.add("http://nothinghere", method=method, status=status_code)

    response = await async_client.request(method, "http://nothinghere")
    assert response.status == status_codes[-1], "response status is not as expected"

    # ensure we called the request the correct number of times
    assert aioresponses_mock.called_times() == len(
        status_codes
    ), "Wrong number of retries"


@pytest.mark.asyncio
async def test_headers_filtering(
    async_client: AsyncClientWithRetry, aioresponses_mock: aioresponses_mock
):
    """
    Header keys/values type must be str to be serializable
    This tests ensures we drop headers with 'None' values
    """

    def callback(url, **kwargs):
        return aioresponses.CallbackResult(headers=kwargs["headers"])

    aioresponses_mock.add("http://nothinghere", method="GET", callback=callback)

    response = await async_client.get(
        "http://nothinghere", headers={"x": None, "y": "z"}
    )
    assert response.headers["y"] == "z", "header should not have been filtered"
    assert "x" not in response.headers, "header with 'None' value was not filtered"


def raise_exception():
    try:
        raise ConnectionError("This is an ErrorA")
    except ConnectionError as e1:
        try:
            raise Exception from e1
        except Exception as e2:
            return e2


@pytest.mark.parametrize(
    "exception,expected",
    [
        # Test error cases that occur twice,
        # and are retryable errors, so we expect no Exception to be raised
        (ConnectionError("This is an ConnectionErr"), does_not_raise()),
        (ConnectionRefusedError("This is an ConnectionRefusedErr"), does_not_raise()),
        (
            ClientConnectorError(mock.MagicMock(code=500), ConnectionResetError()),
            does_not_raise(),
        ),
        # Test a custom exception with a root cause that is included in our retryable exceptions list,
        # should not raise an exception
        (raise_exception(), does_not_raise()),
        # Test a non-retryable error and ensure it fails immediately and is not retried
        (TypeError("TypeErr"), pytest.raises(TypeError)),
    ],
)
@pytest.mark.asyncio
async def test_session_retry(
    async_client: AsyncClientWithRetry,
    aioresponses_mock: aioresponses_mock,
    exception,
    expected,
):
    max_retries = 3
    for i in range(max_retries - 1):
        aioresponses_mock.get(
            "http://localhost:30678",
            exception=exception,
        )
        aioresponses_mock.get(
            "http://localhost:30678",
            status=200,
        )
    with expected:
        await async_client.get("http://localhost:30678")
