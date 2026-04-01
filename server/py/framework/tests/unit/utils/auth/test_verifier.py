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

import asyncio
import time
import unittest.mock
from collections.abc import Generator

import fastapi
import jwt
import pytest
import starlette.datastructures

import mlrun
import mlrun.common.schemas as schemas
import mlrun.utils.singleton

import framework.utils.auth.verifier
import framework.utils.clients.iguazio.v4
from framework.utils.auth.verifier import TOKEN_CACHE_MAX_TTL

# --- Helpers ---


def _make_jwt(exp: float | None = None) -> str:
    payload = {}
    if exp is not None:
        payload["exp"] = int(exp)
    return jwt.encode(payload, key="test-secret", algorithm="HS256")


def _make_request(token: str | None, scheme: str = "Bearer") -> fastapi.Request:
    headers = {}
    if token is not None:
        headers["Authorization"] = f"{scheme} {token}"
    request = fastapi.Request({"type": "http"})
    request._headers = starlette.datastructures.Headers(headers)
    return request


# --- Fixtures ---


@pytest.fixture(autouse=True)
def reset_verifier() -> Generator[None, None, None]:
    """Reset the AuthVerifier singleton before and after each test."""
    original_mode = mlrun.mlconf.httpdb.authorization.mode
    mlrun.mlconf.httpdb.authorization.mode = "none"
    mlrun.utils.singleton.Singleton._instances.pop(
        framework.utils.auth.verifier.AuthVerifier, None
    )
    yield
    mlrun.mlconf.httpdb.authorization.mode = original_mode
    mlrun.utils.singleton.Singleton._instances.pop(
        framework.utils.auth.verifier.AuthVerifier, None
    )


@pytest.fixture
def verifier() -> framework.utils.auth.verifier.AuthVerifier:
    return framework.utils.auth.verifier.AuthVerifier()


@pytest.fixture
def mock_client() -> Generator[
    tuple[unittest.mock.AsyncMock, schemas.AuthInfo], None, None
]:
    """Patches AsyncClient and returns (mock_instance, default_auth_info)."""
    auth_info = schemas.AuthInfo(username="test-user")
    mock_instance = unittest.mock.AsyncMock()
    mock_instance.verify_request_session.return_value = auth_info
    with unittest.mock.patch(
        "framework.utils.clients.iguazio.v4.AsyncClient",
        return_value=mock_instance,
    ):
        yield mock_instance, auth_info


# --- Cache miss / hit ---


@pytest.mark.asyncio
async def test_cache_miss_calls_backend(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    client, auth_info = mock_client
    token = _make_jwt(exp=time.time() + 3600)

    result = await verifier._authenticate_iguazio_v4(_make_request(token))

    assert result == auth_info
    client.verify_request_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_hit_reuses_result(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    client, auth_info = mock_client
    token = _make_jwt(exp=time.time() + 3600)
    request = _make_request(token)

    result1 = await verifier._authenticate_iguazio_v4(request)
    result2 = await verifier._authenticate_iguazio_v4(request)

    assert result1 == auth_info
    assert result2 == auth_info
    # Backend should only be called once despite two requests
    client.verify_request_session.assert_awaited_once()


# --- Scenarios that skip the cache ---


@pytest.mark.asyncio
async def test_no_auth_header_skips_cache(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    await verifier._authenticate_iguazio_v4(_make_request(None))

    assert len(verifier._token_cache) == 0


@pytest.mark.asyncio
async def test_non_bearer_scheme_skips_cache(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    token = _make_jwt(exp=time.time() + 3600)

    await verifier._authenticate_iguazio_v4(_make_request(token, scheme="Basic"))

    assert len(verifier._token_cache) == 0


@pytest.mark.asyncio
async def test_non_jwt_bearer_skips_cache(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    await verifier._authenticate_iguazio_v4(_make_request("not-a-jwt-token"))

    assert len(verifier._token_cache) == 0


@pytest.mark.asyncio
async def test_jwt_without_exp_skips_cache(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    token = _make_jwt(exp=None)

    await verifier._authenticate_iguazio_v4(_make_request(token))

    assert len(verifier._token_cache) == 0


@pytest.mark.asyncio
async def test_expired_jwt_skips_cache(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    token = _make_jwt(exp=time.time() - 1)

    await verifier._authenticate_iguazio_v4(_make_request(token))

    assert len(verifier._token_cache) == 0


# --- Failure and eviction ---


@pytest.mark.asyncio
async def test_backend_failure_evicts_task(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    client, _ = mock_client
    client.verify_request_session.side_effect = Exception("backend unavailable")
    token = _make_jwt(exp=time.time() + 3600)
    request = _make_request(token)

    with pytest.raises(Exception, match="backend unavailable"):
        await verifier._authenticate_iguazio_v4(request)

    # Done callbacks are scheduled via call_soon; yield to let them run
    await asyncio.sleep(0)

    assert token not in verifier._token_cache

    # The next request should retry rather than returning the failed task
    client.verify_request_session.side_effect = None
    client.verify_request_session.return_value = schemas.AuthInfo(username="retry-user")
    await verifier._authenticate_iguazio_v4(request)
    assert client.verify_request_session.call_count == 2


@pytest.mark.asyncio
async def test_lru_eviction(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(framework.utils.auth.verifier, "TOKEN_CACHE_MAX_SIZE", 2)

    exp = time.time() + 3600
    # Three distinct tokens (different exp → different JWT payloads)
    tokens = [_make_jwt(exp=exp + i) for i in range(3)]

    for token in tokens:
        await verifier._authenticate_iguazio_v4(_make_request(token))

    assert tokens[0] not in verifier._token_cache
    assert tokens[1] in verifier._token_cache
    assert tokens[2] in verifier._token_cache


@pytest.mark.asyncio
async def test_ttl_expiry(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    client, _ = mock_client
    base_time = time.time()
    token = _make_jwt(exp=base_time + 3600)
    request = _make_request(token)

    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        mock_time.time.return_value = base_time
        await verifier._authenticate_iguazio_v4(request)

    assert token in verifier._token_cache

    # Advance time past TTL; _authenticate_iguazio_v4 should expire the token
    # internally and call the backend again
    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        mock_time.time.return_value = base_time + TOKEN_CACHE_MAX_TTL + 1
        await verifier._authenticate_iguazio_v4(request)

    assert client.verify_request_session.call_count == 2


# --- Concurrency ---


@pytest.mark.asyncio
async def test_concurrent_requests_share_single_backend_call(
    verifier: framework.utils.auth.verifier.AuthVerifier,
):
    token = _make_jwt(exp=time.time() + 3600)
    auth_info = schemas.AuthInfo(username="test-user")

    backend_started = asyncio.Event()
    backend_proceed = asyncio.Event()

    async def slow_verify(_request):
        backend_started.set()
        await backend_proceed.wait()
        return auth_info

    mock_instance = unittest.mock.AsyncMock()
    mock_instance.verify_request_session.side_effect = slow_verify

    with unittest.mock.patch(
        "framework.utils.clients.iguazio.v4.AsyncClient",
        return_value=mock_instance,
    ):
        # Start first request and wait until the backend call is in-flight
        task1 = asyncio.create_task(
            verifier._authenticate_iguazio_v4(_make_request(token))
        )
        await backend_started.wait()

        # Start second request while the first is still waiting on the backend
        task2 = asyncio.create_task(
            verifier._authenticate_iguazio_v4(_make_request(token))
        )
        backend_proceed.set()

        result1, result2 = await asyncio.gather(task1, task2)

    assert result1 == auth_info
    assert result2 == auth_info
    mock_instance.verify_request_session.assert_awaited_once()


# --- Heap expiry paths ---


@pytest.mark.asyncio
async def test_rebuild_path_expires_valid_token(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
    monkeypatch: pytest.MonkeyPatch,
):
    """
    The rebuild path (heap > 2*cache) is triggered through _authenticate_iguazio_v4
    and correctly expires a token that is still in cache but past its TTL.

    Setup with max_size=1:
      t=0: A cached → B cached (evicts A) → A re-cached (evicts B)
      Result: heap has 3 entries [A_v1, B, A_v2], cache has 1 entry {A: A_v2}

    At t=TTL+1, adding C triggers _expire_tokens with heap=3 > cache=1*2:
      - A_v1: stale (different task)  → skip
      - B:    stale (not in cache)    → skip
      - A_v2: valid but expired       → deleted  ← the branch under test
    """
    monkeypatch.setattr(framework.utils.auth.verifier, "TOKEN_CACHE_MAX_SIZE", 1)
    client, _ = mock_client

    base_time = time.time()
    token_a = _make_jwt(exp=base_time + 3600)
    token_b = _make_jwt(exp=base_time + 3601)
    token_c = _make_jwt(exp=base_time + 3602)

    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        mock_time.time.return_value = base_time
        await verifier._authenticate_iguazio_v4(_make_request(token_a))
        await verifier._authenticate_iguazio_v4(_make_request(token_b))
        await verifier._authenticate_iguazio_v4(_make_request(token_a))

    # Adding token_c at TTL+1 triggers the rebuild; token_a_v2 is expired and deleted
    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        mock_time.time.return_value = base_time + TOKEN_CACHE_MAX_TTL + 1
        await verifier._authenticate_iguazio_v4(_make_request(token_c))

    # a_v1, b, a_v2 at t=0, then c at t=TTL+1
    assert client.verify_request_session.call_count == 4
    assert token_c in verifier._token_cache
    assert token_a not in verifier._token_cache


@pytest.mark.asyncio
async def test_fast_path_skips_stale_heap_entry(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
    monkeypatch: pytest.MonkeyPatch,
):
    """
    In the fast path (heap <= 2*cache), an expired heap entry whose token has
    already been LRU-evicted is silently skipped rather than causing an error.

    Setup with max_size=1:
      t=0: A cached → B cached (evicts A)
      Result: heap=[A(TTL), B(TTL)], cache={B: task_b}; heap=2, cache=1 → 2 <= 2, fast path

    At t=TTL+1, requesting A again triggers fast path expiry:
      - A's entry: `cache.get(A) is task_a` → False (A was evicted) → no-op  ← the branch under test
      - B's entry: `cache.get(B) is task_b` → True                → evicted
    """
    monkeypatch.setattr(framework.utils.auth.verifier, "TOKEN_CACHE_MAX_SIZE", 1)
    client, _ = mock_client

    base_time = time.time()
    token_a = _make_jwt(exp=base_time + 3600)
    token_b = _make_jwt(exp=base_time + 3601)

    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        mock_time.time.return_value = base_time
        await verifier._authenticate_iguazio_v4(_make_request(token_a))
        await verifier._authenticate_iguazio_v4(_make_request(token_b))

    # At TTL+1, re-requesting A triggers fast path; A's stale entry is skipped,
    # B is evicted, and A gets a fresh backend call
    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        mock_time.time.return_value = base_time + TOKEN_CACHE_MAX_TTL + 1
        await verifier._authenticate_iguazio_v4(_make_request(token_a))

    # a (t=0), b (t=0), a_v2 (t=TTL+1)
    assert client.verify_request_session.call_count == 3
    assert token_a in verifier._token_cache
    assert token_b not in verifier._token_cache


# --- Stale heap entry regression ---


@pytest.mark.asyncio
async def test_stale_heap_entry_doesnt_evict_recached_token(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Regression test for the double-heap-entry bug:

    1. Token A is cached (heap entry 1 with task_a_v1, expires at base+TTL).
    2. Token B is cached, LRU-evicting token A. Heap entry 1 is now stale.
    3. Token A is re-cached (heap entry 2 with task_a_v2, expires at base+5+TTL).
    4. At base+TTL+1, heap entry 1 fires. It must NOT evict task_a_v2 from cache.
    """
    monkeypatch.setattr(framework.utils.auth.verifier, "TOKEN_CACHE_MAX_SIZE", 1)

    base_time = time.time()
    token_a = _make_jwt(exp=base_time + 3600)
    token_b = _make_jwt(exp=base_time + 3601)

    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        # t=0: cache token_a → heap entry 1 (expires at base_time + TTL)
        mock_time.time.return_value = base_time
        await verifier._authenticate_iguazio_v4(_make_request(token_a))
        task_a_v1 = verifier._token_cache[token_a]

        # t=0: cache token_b → LRU-evicts token_a; heap entry 1 becomes stale
        await verifier._authenticate_iguazio_v4(_make_request(token_b))
        assert token_a not in verifier._token_cache

        # t=5: re-cache token_a → heap entry 2 (expires at base_time+5+TTL)
        mock_time.time.return_value = base_time + 5
        await verifier._authenticate_iguazio_v4(_make_request(token_a))
        task_a_v2 = verifier._token_cache[token_a]
        assert task_a_v2 is not task_a_v1

    # t=TTL+1: heap entry 1 (base_time+TTL) fires but entry 2 (base_time+5+TTL)
    # is still valid. The stale entry must not evict task_a_v2.
    verifier._expire_tokens(base_time + TOKEN_CACHE_MAX_TTL + 1)

    assert token_a in verifier._token_cache, (
        "task_a_v2 should still be cached after stale heap entry fires"
    )
    assert verifier._token_cache[token_a] is task_a_v2
