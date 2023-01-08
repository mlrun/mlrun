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
import threading
from unittest import mock

import pytest
from aiohttp import (
    ClientConnectorError,
    ClientResponse,
    ClientResponseError,
    ServerDisconnectedError,
)

from mlrun.errors import MLRunHTTPError
from mlrun.errors import raise_for_status as mlrun_raise_for_status
from mlrun.utils.async_http import AsyncClientWithRetry
from mlrun.utils.logger import create_logger


@pytest.fixture
async def async_client():
    async with AsyncClientWithRetry() as client:
        client._client.request = mock.AsyncMock()
        return client


@pytest.fixture
def httpd_on_demand():
    httpd_on_demand = HTTPDaemonOnDemand()
    yield httpd_on_demand
    httpd_on_demand.close()


@pytest.mark.asyncio
async def test_retry_exceptions(httpd_on_demand: "HTTPDaemonOnDemand"):
    retry_counter = 0

    def wrap_real_request(client_):
        def handle_request(*args, **kwargs):
            nonlocal retry_counter
            retry_counter += 1
            return real_handle_request(*args, **kwargs)

        real_handle_request = client_._client.request
        client_._client.request = handle_request

    async with AsyncClientWithRetry(
        read_timeout=10,
        conn_timeout=10,
    ) as client:
        wrap_real_request(client)

        # Case - Server was not started, expect it to refuse our connection
        with pytest.raises(ClientConnectorError) as exc:
            await client.get(f"http://localhost:{httpd_on_demand.port}")
        assert retry_counter == 3, "should have retried 3 times"
        assert exc.value.os_error.errno == 61, "expected connection refused"

        # starting the server
        httpd_on_demand.start()

        # Case - Server is started, but it's not responding, or closing the connection prematurely
        # reset counter
        retry_counter = 0
        with pytest.raises(ServerDisconnectedError) as exc:
            await client.get(
                f"http://localhost:{httpd_on_demand.port}",
                headers={
                    "x-action": "close",
                },
            )
        assert retry_counter == 3, "should have retried 3 times"
        assert exc.value.message == "Server disconnected"

        # Case - Server is started, but it's not responding, or closing the connection prematurely
        # Do not retry because method is blacklisted
        # reset counter
        retry_counter = 0
        client.retry_options.blacklisted_methods = ["POST"]
        with pytest.raises(ServerDisconnectedError) as exc:
            await client.post(
                f"http://localhost:{httpd_on_demand.port}",
                headers={
                    "x-action": "close",
                },
            )
        assert (
            retry_counter == 1
        ), "should have retried 1 time, POST is a blacklisted method"
        assert exc.value.message == "Server disconnected"


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
async def test_retry_method_status_codes(async_client, method, status_codes):
    def dummy_raise_for_status():
        status_code = status_codes[async_client._client.request.call_count - 1]
        if status_code >= http.HTTPStatus.BAD_REQUEST:
            raise ClientResponseError(
                mock.MagicMock(),
                mock.MagicMock(),
                # return the last status code from the retry list
                status=status_codes[async_client._client.request.call_count - 1],
            )

    async_client._client.request.side_effect = [
        mock.AsyncMock(
            spec=ClientResponse,
            status=status_code,
            raise_for_status=dummy_raise_for_status,
        )
        for status_code in status_codes
    ]
    response = await async_client.request(method, "http://nothinghere")
    assert response.status == status_codes[-1], "response status is not as expected"

    # ensure we called the request the correct number of times
    assert async_client._client.request.call_count == len(
        status_codes
    ), "Wrong number of retries"


class HTTPDaemonOnDemand:
    def __init__(self):
        self._logger = create_logger("DEBUG", name="async-http-logger")
        self._port = 30666
        self._httpd = None
        self._httpd_thread = None
        self._started = False
        self._closed = False

    @property
    def httpd(self):
        return self._httpd

    @property
    def port(self):
        return self._port

    def start(self):
        self._logger.debug("Starting HTTP daemon")
        self._httpd = http.server.HTTPServer(
            ("localhost", self._port), HTTPDaemonOnDemandRequestHandler
        )
        self._httpd_thread = threading.Thread(target=self.run_server)
        self._httpd_thread.start()
        self._started = True

    def run_server(self):
        self._logger.debug("Running HTTP daemon")
        self._httpd.serve_forever()

    def close(self):
        if self._closed:
            return
        self._logger.debug("Closing HTTP daemon")
        self._httpd.shutdown()
        self._logger.debug("HTTP daemon closed, joining thread")
        self._httpd_thread.join()
        self._closed = True


class HTTPDaemonOnDemandRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self._default_handler()

    def do_POST(self):
        self._default_handler()

    def _default_handler(self):
        action = self.headers.get("x-action", "")
        status_code = int(self.headers.get("x-status-code", http.HTTPStatus.OK))
        if action == "close":
            self.request.close()
        else:
            self.send_response(status_code)
            self.end_headers()
