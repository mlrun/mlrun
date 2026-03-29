# Copyright 2026 Iguazio
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

import unittest.mock

import pytest
import requests_mock as requests_mock_module

import mlrun.common.types
import mlrun.db.httpdb
import mlrun.utils.http

http_methods = [method.value for method in mlrun.common.types.HTTPMethod]


@pytest.mark.parametrize("method", http_methods)
def test_calls_reuse_same_session(method: str):
    """{method} calls should reuse the existing session, not create a new one."""
    db = mlrun.db.httpdb.HTTPRunDB("https://fake-url")

    with requests_mock_module.Mocker() as adapter:
        adapter.register_uri(method, "https://fake-url/api/v1/some/path", json={})

        db.api_call(method, "some/path")
        first_session = db.session

        db.api_call(method, "some/path")
        second_session = db.session

        assert first_session is second_session, (
            f"{method} should reuse the existing session, not create a new one"
        )


def test_mixed_methods_reuse_same_session():
    """GET, POST, PUT calls should all share the same session object."""
    db = mlrun.db.httpdb.HTTPRunDB("https://fake-url")

    with requests_mock_module.Mocker() as adapter:
        adapter.register_uri("GET", "https://fake-url/api/v1/some/path", json={})
        adapter.register_uri("POST", "https://fake-url/api/v1/some/path", json={})
        adapter.register_uri("PUT", "https://fake-url/api/v1/some/path", json={})

        db.api_call("GET", "some/path")
        session_after_get = db.session

        db.api_call("POST", "some/path")
        session_after_post = db.session

        db.api_call("PUT", "some/path")
        session_after_put = db.session

        assert session_after_get is session_after_post is session_after_put, (
            "All HTTP methods should share the same session object"
        )


def test_multiple_post_calls_no_session_leak():
    """N POST calls should create exactly 1 session, not N."""
    db = mlrun.db.httpdb.HTTPRunDB("https://fake-url")
    sessions_created = []

    original_init_session = db._init_session

    def tracking_init_session(*args, **kwargs):
        session = original_init_session(*args, **kwargs)
        sessions_created.append(session)
        return session

    db._init_session = tracking_init_session

    with requests_mock_module.Mocker() as adapter:
        adapter.register_uri("POST", "https://fake-url/api/v1/some/path", json={})

        num_calls = 5
        for _ in range(num_calls):
            db.api_call("POST", "some/path")

        assert len(sessions_created) == 1, (
            f"Expected 1 session created for {num_calls} POST calls, "
            f"got {len(sessions_created)}"
        )


@pytest.mark.parametrize(
    "retry_on_post, retry_on_put",
    [
        (True, False),
        (False, True),
        (True, True),
        (False, False),
    ],
)
def test_update_retry_methods_updates_frozenset(retry_on_post, retry_on_put):
    """update_retry_methods() should update _retry_methods on the session."""
    session = mlrun.utils.http.HTTPSessionWithRetry(
        retry_on_post=False, retry_on_put=False
    )

    session.update_retry_methods(retry_on_post=retry_on_post, retry_on_put=retry_on_put)

    assert ("POST" in session._retry_methods) == retry_on_post
    assert ("PUT" in session._retry_methods) == retry_on_put


def test_update_retry_methods_updates_adapter():
    """update_retry_methods() should update the adapter's Retry.allowed_methods."""
    session = mlrun.utils.http.HTTPSessionWithRetry(
        retry_on_post=False, retry_on_put=True, retry_on_status=True
    )

    assert "POST" not in session._http_adapter.max_retries.allowed_methods

    session.update_retry_methods(retry_on_post=True, retry_on_put=True)

    assert "POST" in session._http_adapter.max_retries.allowed_methods
    assert "PUT" in session._http_adapter.max_retries.allowed_methods


def test_update_retry_methods_without_adapter():
    """update_retry_methods() should work when retry_on_status=False (no adapter)."""
    session = mlrun.utils.http.HTTPSessionWithRetry(
        retry_on_post=False, retry_on_put=True, retry_on_status=False
    )

    assert not hasattr(session, "_http_adapter")

    # Should not raise
    session.update_retry_methods(retry_on_post=True, retry_on_put=False)

    assert "POST" in session._retry_methods
    assert "PUT" not in session._retry_methods


def test_update_retry_methods_called_on_post_api_call():
    """api_call() should call update_retry_methods() on POST when session exists."""
    db = mlrun.db.httpdb.HTTPRunDB("https://fake-url")

    with requests_mock_module.Mocker() as adapter:
        adapter.register_uri("GET", "https://fake-url/api/v1/some/path", json={})
        adapter.register_uri("POST", "https://fake-url/api/v1/some/path", json={})

        # Create session via GET
        db.api_call("GET", "some/path")
        assert db.session is not None

        # Spy on update_retry_methods
        db.session.update_retry_methods = unittest.mock.Mock(
            wraps=db.session.update_retry_methods
        )

        # POST should call update_retry_methods instead of replacing session
        db.api_call("POST", "some/path")

        db.session.update_retry_methods.assert_called_once()
