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
import json
import socketserver
import threading
import time
import unittest
from contextlib import nullcontext as does_not_raise

import pytest

from mlrun.utils.http import HTTPSessionWithRetry


@pytest.fixture
def http_session():
    with HTTPSessionWithRetry() as session:
        yield session


def raise_exception():
    try:
        raise ConnectionError("This is an ErrorA")
    except ConnectionError as e1:
        try:
            raise Exception from e1
        except Exception as e2:
            return e2


@pytest.mark.parametrize(
    "error_to_raise,expected",
    [
        # Test ConnectionError and ConnectionRefusedError cases that occur once,
        # and are retryable errors, so we expect no Exception to be raised
        ([ConnectionError("This is an ConnectionErr"), True], does_not_raise()),
        (
            [ConnectionRefusedError("This is a ConnectionRefusedErr"), True],
            does_not_raise(),
        ),
        # Test a custom exception with a root cause that is included in our retryable exceptions list,
        # should not raise an exception
        ([raise_exception(), True], does_not_raise()),
        # Test a custom exception with a root cause that is included in our retryable exceptions list,
        # it will be raised 3 times before we expect an error to be raised
        (raise_exception(), pytest.raises(Exception)),
        # Test a non-retryable error and ensure it fails immediately and is not retried
        ([TypeError("TypeErr"), True], pytest.raises(TypeError)),
    ],
)
def test_session_retry(http_session: HTTPSessionWithRetry, error_to_raise, expected):
    with unittest.mock.patch(
        "mlrun.utils.http.requests.Session.request", side_effect=error_to_raise
    ):
        with expected:
            http_session.request("GET", "http://localhost:30678")


def test_http_session_does_not_persist_cookies():
    """Test that HTTPSessionWithRetry doesn't store or send cookies"""

    class MockServer(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            cookie_header = self.headers.get("Cookie", "")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Set-Cookie", "test_cookie=test_value; Path=/")
            self.end_headers()
            self.wfile.write(json.dumps({"cookies_received": cookie_header}).encode())

    # Use ephemeral port (OS-chosen)
    server = socketserver.TCPServer(("127.0.0.1", 0), MockServer)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)  # Minimal sleep for server readiness

    try:
        session = HTTPSessionWithRetry()

        # Request 1 - server sets cookie
        resp1 = session.get(f"http://127.0.0.1:{port}/first")
        # Check what cookies the server RECEIVED from us (should be none)
        assert resp1.json()["cookies_received"] == ""

        # Check what cookies the server SENT to us (in response)
        # We CAN read cookies from the response, they're just not stored for future requests
        assert "Set-Cookie" in resp1.headers, "Server should send Set-Cookie header"
        assert "test_cookie=test_value" in resp1.headers["Set-Cookie"]

        # Request 2 - verify cookie was NOT sent back to server
        resp2 = session.get(f"http://127.0.0.1:{port}/second")
        # This verifies the cookie from Request 1 was NOT sent back
        # (proves DummyCookieJar didn't store it)
        assert (
            resp2.json()["cookies_received"] == ""
        ), "HTTPSessionWithRetry should not persist cookies"

        # Server still sets cookies in response, and we can still read them
        # But they won't be stored or sent in future requests
        assert "Set-Cookie" in resp2.headers, "Server still sends cookies in response"

        # Verify cookie jar is empty (cookies were not stored)
        assert len(session.cookies) == 0, "Cookie jar should be empty"

        session.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)
