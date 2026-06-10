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

"""Unit tests for ``mlrun.Client``."""

from __future__ import annotations

import unittest.mock

import pytest

import mlrun
import mlrun.errors
from mlrun import Client, Credentials


@pytest.fixture(autouse=True)
def _mock_dbpath(monkeypatch):
    monkeypatch.setattr(mlrun.mlconf, "dbpath", "https://mock-server")


def test_session_routes_get_run_db_to_per_client_http_db():
    """Inside ``session()``, ``get_run_db()`` returns the client's HTTPRunDB."""
    client_a = Client(credentials=Credentials(token="token-a"))
    client_b = Client(credentials=Credentials(token="token-b"))

    with client_a.session():
        assert mlrun.get_run_db() is client_a._http_db

    with client_b.session():
        assert mlrun.get_run_db() is client_b._http_db

    # Outside any session: legacy singleton.
    legacy_db = mlrun.get_run_db()
    assert legacy_db is not client_a._http_db
    assert legacy_db is not client_b._http_db


def test_session_carries_client_credentials_to_requests():
    """Requests carry the client's bearer token, not process env."""
    client = Client(credentials=Credentials(token="my-token"))
    client._http_db.session = unittest.mock.Mock()

    with client.session():
        mlrun.get_run_db().api_call("GET", "some-path")

    request_kwargs = client._http_db.session.request.call_args[1]
    headers = request_kwargs.get("headers", {})
    # MLRun uses lowercase "authorization" (see mlrun.common.schemas.HeaderNames).
    assert headers.get("authorization") == "Bearer my-token"


def test_session_carries_extra_headers_to_requests():
    """``Credentials.extra_headers`` are added to each request."""
    client = Client(
        credentials=Credentials(
            token="my-token", extra_headers={"X-IGZ-Authenticator-Kind": "sa"}
        )
    )
    client._http_db.session = unittest.mock.Mock()

    with client.session():
        mlrun.get_run_db().api_call("GET", "some-path")

    headers = client._http_db.session.request.call_args[1].get("headers", {})
    assert headers.get("X-IGZ-Authenticator-Kind") == "sa"
    assert headers.get("authorization") == "Bearer my-token"


def test_per_call_header_overrides_default_extra_header():
    """Per-call ``headers=`` values override ``Credentials`` defaults."""
    client = Client(
        credentials=Credentials(
            token="my-token", extra_headers={"X-IGZ-Authenticator-Kind": "sa"}
        )
    )
    client._http_db.session = unittest.mock.Mock()

    with client.session():
        mlrun.get_run_db().api_call(
            "GET", "some-path", headers={"X-IGZ-Authenticator-Kind": "override"}
        )

    headers = client._http_db.session.request.call_args[1].get("headers", {})
    assert headers.get("X-IGZ-Authenticator-Kind") == "override"


def test_extra_header_cannot_override_authorization():
    """``extra_headers`` must not override the auth header."""
    client = Client(
        credentials=Credentials(
            token="my-token", extra_headers={"Authorization": "Bearer spoofed"}
        )
    )
    client._http_db.session = unittest.mock.Mock()

    with client.session():
        mlrun.get_run_db().api_call("GET", "some-path")

    headers = client._http_db.session.request.call_args[1].get("headers", {})
    assert headers.get("authorization") == "Bearer my-token"


def test_credentials_use_env_matches_legacy_singleton_auth(monkeypatch):
    """``Credentials(use_env=True)`` resolves auth like the legacy singleton."""
    monkeypatch.setenv("V3IO_ACCESS_KEY", "host-process-token")

    legacy = mlrun.db.httpdb.HTTPRunDB("https://mock-server")
    client = Client(credentials=Credentials(use_env=True))

    # Same provider class, same captured token.
    assert type(client._http_db.token_provider) is type(legacy.token_provider)
    assert (
        client._http_db.token_provider.get_token() == legacy.token_provider.get_token()
    )


def test_credentials_with_basic_auth_sets_user_password():
    """``username=/password=`` configures HTTP basic auth."""
    client = Client(credentials=Credentials(username="alice", password="secret"))

    assert client._http_db.user == "alice"
    assert client._http_db.password == "secret"
    assert client._http_db.token_provider is None


def test_credentials_empty_is_rejected():
    """Bare ``Credentials()`` has no auth mode and must error fast."""
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        Credentials()


def test_credentials_extra_headers_only_is_rejected():
    """``extra_headers`` alone is not a valid auth mode."""
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        Credentials(extra_headers={"X-IGZ-Authenticator-Kind": "sa"})


def test_credentials_mixed_modes_are_rejected():
    """Combining multiple auth modes must error fast."""
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        Credentials(token="t", username="u", password="p")
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        Credentials(token="t", use_env=True)


def test_credentials_partial_basic_auth_is_rejected():
    """Basic auth with only username or only password must error fast."""
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        Credentials(username="alice")
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        Credentials(password="secret")


def test_nested_sessions_restore_outer_client_on_exit():
    """Nested ``session()`` blocks restore the outer client on inner exit."""
    outer = Client(credentials=Credentials(token="outer"))
    inner = Client(credentials=Credentials(token="inner"))

    with outer.session():
        assert mlrun.get_run_db() is outer._http_db
        with inner.session():
            assert mlrun.get_run_db() is inner._http_db
        assert mlrun.get_run_db() is outer._http_db
