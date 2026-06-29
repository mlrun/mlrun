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

import copy
import http.server
import json
import pickle
import socketserver
import threading
import time
import unittest
from contextlib import nullcontext as does_not_raise

import pytest

from mlrun.utils.http import HTTPSessionWithRetry


def _deepcopy(session: HTTPSessionWithRetry) -> HTTPSessionWithRetry:
    return copy.deepcopy(session)


def _pickle_roundtrip(session: HTTPSessionWithRetry) -> HTTPSessionWithRetry:
    return pickle.loads(pickle.dumps(session))


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
        assert resp2.json()["cookies_received"] == "", (
            "HTTPSessionWithRetry should not persist cookies"
        )

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


@pytest.mark.parametrize(
    "clone",
    [_deepcopy, _pickle_roundtrip],
    ids=["deepcopy", "pickle"],
)
def test_session_survives_copy(http_session: HTTPSessionWithRetry, clone):
    """Regression for ML-12648.

    ``requests.Session`` only serializes its ``__attrs__``, so before the fix a
    ``copy.deepcopy``/``pickle`` of the session dropped the retry attributes set
    in ``__init__``. Reusing such a session for a POST/PUT then called
    ``update_retry_methods`` and raised
    ``AttributeError: ... has no attribute '_retry_methods'``.
    """
    restored = clone(http_session)

    # Attributes set in __init__ that requests.Session would otherwise drop.
    assert restored._retry_methods == http_session._retry_methods
    assert restored.max_retries == http_session.max_retries
    assert restored.retry_backoff_factor == http_session.retry_backoff_factor
    assert restored.retry_on_exception == http_session.retry_on_exception
    assert restored.verbose == http_session.verbose
    # _logger is excluded from the serialized state and rebuilt on restore.
    assert restored._logger is not None

    # The symptom path: reusing the session for a retriable POST must not raise.
    restored.update_retry_methods(retry_on_post=True, retry_on_put=True)
    assert "POST" in restored._retry_methods


@pytest.mark.parametrize(
    "clone",
    [_deepcopy, _pickle_roundtrip],
    ids=["deepcopy", "pickle"],
)
def test_copied_session_has_independent_connection_pool(
    http_session: HTTPSessionWithRetry, clone
):
    """A copy must not share the original's urllib3 connection pool.

    Sharing would regress the socket-leak fix (#9515) that motivated reusing a
    single session - clones would keep the original's sockets alive.
    """
    restored = clone(http_session)

    # Both schemes stay mounted on one adapter and _http_adapter points at it -
    # update_retry_methods relies on this identity to reconfigure retries.
    assert restored.adapters["http://"] is restored.adapters["https://"]
    assert restored._http_adapter is restored.adapters["https://"]
    # ...but the pool itself is a fresh object, not shared with the original.
    assert (
        restored._http_adapter.poolmanager is not http_session._http_adapter.poolmanager
    )


def test_update_retry_methods_after_copy_reconfigures_adapter(
    http_session: HTTPSessionWithRetry,
):
    """After a copy, changing the retry policy must reconfigure the mounted
    adapter's ``Retry``, not only the internal ``_retry_methods`` frozenset."""
    restored = copy.deepcopy(http_session)

    # Default session retries PUT but not POST; drop both to force a change.
    restored.update_retry_methods(retry_on_post=False, retry_on_put=False)

    allowed_methods = restored._http_adapter.max_retries.allowed_methods
    assert "POST" not in allowed_methods
    assert "PUT" not in allowed_methods
    assert restored._retry_methods == allowed_methods


@pytest.mark.parametrize(
    "clone",
    [_deepcopy, _pickle_roundtrip],
    ids=["deepcopy", "pickle"],
)
def test_copy_without_status_retry_keeps_http_adapter_absent(clone):
    """``_http_adapter`` is only set when ``retry_on_status`` is enabled.

    A copy must preserve its *absence* - otherwise the ``hasattr`` guard in
    ``update_retry_methods`` would treat a restored ``None`` as a real adapter
    and raise. This is the path used by serving with retries disabled.
    """
    session = HTTPSessionWithRetry(retry_on_status=False)
    assert "_http_adapter" not in session.__dict__

    restored = clone(session)

    assert not hasattr(restored, "_http_adapter")
    # Must not raise even though there is no adapter to reconfigure.
    restored.update_retry_methods(retry_on_post=True, retry_on_put=True)
    assert "POST" in restored._retry_methods
