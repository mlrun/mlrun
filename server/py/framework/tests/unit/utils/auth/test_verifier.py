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
import unittest.mock
from collections.abc import Generator

import fastapi
import pytest
import starlette.datastructures

import mlrun
import mlrun.common.schemas as schemas
import mlrun.utils.singleton

import framework.utils.auth.verifier
import framework.utils.clients.iguazio.v4
from framework.utils.auth.verifier import TOKEN_CACHE_MAX_TTL

# --- Helpers ---


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
    token = "token"

    result = await verifier._authenticate_iguazio_v4(_make_request(token))

    assert result == auth_info
    client.verify_request_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_hit_reuses_result(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    client, auth_info = mock_client
    token = "token"
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
    await verifier._authenticate_iguazio_v4(_make_request("token", scheme="Basic"))

    assert len(verifier._token_cache) == 0


# --- Failure and eviction ---


@pytest.mark.asyncio
async def test_backend_failure_evicts_task(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    client, _ = mock_client
    client.verify_request_session.side_effect = Exception("backend unavailable")
    token = "token"
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

    tokens = ["token_0", "token_1", "token_2"]

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
    base_time = 0
    token = "token"
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
    token = "token"
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


# --- Stale callback regression ---


@pytest.mark.asyncio
async def test_stale_done_callback_doesnt_evict_refreshed_task(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    """
    When a cached task's TTL expires and is lazily replaced by a new task,
    the old task's done callback must not evict the new task from the cache.

    1. Token A is cached (task_v1 starts but does not complete yet).
    2. At t=TTL+1, token A is requested again; lazy expiry replaces task_v1 with task_v2.
    3. task_v1 fails; its done callback fires.
    4. task_v2 must still be in cache.
    """
    client, _ = mock_client
    base_time = 0
    token = "token"

    backend_proceed = asyncio.Event()
    call_count = 0

    async def controlled_verify(_request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await backend_proceed.wait()
            raise Exception("old task failed")
        return schemas.AuthInfo(username="new-user")

    client.verify_request_session.side_effect = controlled_verify

    # Start first request at t=0; task_v1 blocks waiting on backend_proceed
    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        mock_time.time.return_value = base_time
        task_outer = asyncio.create_task(
            verifier._authenticate_iguazio_v4(_make_request(token))
        )
        await asyncio.sleep(0)  # yield to let task_v1 start

    # At t=TTL+1, lazy expiry fires; task_v2 is created and returned
    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        mock_time.time.return_value = base_time + TOKEN_CACHE_MAX_TTL + 1
        result = await verifier._authenticate_iguazio_v4(_make_request(token))

    assert result.username == "new-user"

    # Release task_v1 to fail; its done callback must not evict task_v2
    backend_proceed.set()
    with pytest.raises(Exception, match="old task failed"):
        await task_outer
    await asyncio.sleep(0)  # let the done callback run

    assert token in verifier._token_cache, (
        "task_v2 should still be cached after stale task_v1 callback fires"
    )
