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

"""Unit tests for ``mlrun.Client``.

``"mock-server"`` in the URL short-circuits ``HTTPRunDB.connect()``.
"""

from __future__ import annotations

import unittest.mock

import mlrun
import mlrun.auth
from mlrun import Client, Credentials


def test_session_routes_get_run_db_to_per_client_http_db():
    """Inside ``session()``, ``get_run_db()`` returns the client's HTTPRunDB."""
    client_a = Client(
        dbpath="https://mock-server-a",
        credentials=Credentials(token="token-a"),
    )
    client_b = Client(
        dbpath="https://mock-server-b",
        credentials=Credentials(token="token-b"),
    )

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
    client = Client(
        dbpath="https://mock-server",
        credentials=Credentials(token="my-token"),
    )
    client._http_db.session = unittest.mock.Mock()

    with client.session():
        mlrun.get_run_db().api_call("GET", "some-path")

    request_kwargs = client._http_db.session.request.call_args[1]
    headers = request_kwargs.get("headers", {})
    # MLRun uses lowercase "authorization" (see mlrun.common.schemas.HeaderNames).
    assert headers.get("authorization") == "Bearer my-token"


def test_credentials_from_env_matches_legacy_singleton_auth(monkeypatch):
    """``Credentials.from_env()`` resolves auth like the legacy singleton."""
    monkeypatch.setenv("V3IO_ACCESS_KEY", "host-process-token")

    legacy = mlrun.db.httpdb.HTTPRunDB("https://mock-server")
    client = Client(
        dbpath="https://mock-server",
        credentials=Credentials.from_env(),
    )

    # Same provider class, same captured token.
    assert type(client._http_db.token_provider) is type(legacy.token_provider)
    assert (
        client._http_db.token_provider.get_token() == legacy.token_provider.get_token()
    )


def test_credentials_with_token_provider_uses_it_directly():
    """A pre-built ``TokenProvider`` is installed on the HTTPRunDB as-is."""
    provider = mlrun.auth.StaticTokenProvider("provider-token")
    client = Client(
        dbpath="https://mock-server",
        credentials=Credentials(token_provider=provider),
    )

    assert client._http_db.token_provider is provider


def test_credentials_with_basic_auth_sets_user_password():
    """``username=/password=`` configures HTTP basic auth."""
    client = Client(
        dbpath="https://mock-server",
        credentials=Credentials(username="alice", password="secret"),
    )

    assert client._http_db.user == "alice"
    assert client._http_db.password == "secret"
    assert client._http_db.token_provider is None


def test_nested_sessions_restore_outer_client_on_exit():
    """Nested ``session()`` blocks restore the outer client on inner exit."""
    outer = Client(
        dbpath="https://mock-server-outer",
        credentials=Credentials(token="outer"),
    )
    inner = Client(
        dbpath="https://mock-server-inner",
        credentials=Credentials(token="inner"),
    )

    with outer.session():
        assert mlrun.get_run_db() is outer._http_db
        with inner.session():
            assert mlrun.get_run_db() is inner._http_db
        assert mlrun.get_run_db() is outer._http_db
