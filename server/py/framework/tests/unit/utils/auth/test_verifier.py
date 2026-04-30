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

import framework.utils.auth.verifier
import framework.utils.clients.iguazio.v4


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


@pytest.mark.asyncio
async def test_cache_miss_calls_backend(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    client, auth_info = mock_client
    token = _make_jwt(exp=time.time() + 3600)

    result = await verifier._authenticate_iguazio_v4(_make_request(token))

    assert result.username == auth_info.username
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

    assert result1.username == auth_info.username
    assert result2.username == auth_info.username
    # Backend should only be called once despite two requests
    client.verify_request_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_returned_auth_info_is_isolated_from_cache(
    verifier: framework.utils.auth.verifier.AuthVerifier,
):
    """Mutations to the returned auth_info must not affect the cached copy.

    authenticate_request mutates the returned auth_info after the fact
    (e.g. sets request_headers). The cache must be shielded from those changes
    so that subsequent callers always get a clean copy.
    """
    token = _make_jwt(exp=time.time() + 3600)
    mock_instance = unittest.mock.AsyncMock()
    mock_instance.verify_request_session.return_value = schemas.AuthInfo(
        username="test-user"
    )

    with unittest.mock.patch(
        "framework.utils.clients.iguazio.v4.AsyncClient",
        return_value=mock_instance,
    ):
        result = await verifier._authenticate_iguazio_v4(_make_request(token))

    # Simulate what authenticate_request does: mutate the returned auth_info
    result.request_headers = {"Authorization": f"Bearer {token}"}

    cached_task, _ = verifier._token_cache[
        framework.utils.auth.verifier.AuthVerifier._token_cache_key(token)
    ]
    cached_auth_info = await cached_task
    assert cached_auth_info.request_headers is None


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


@pytest.mark.asyncio
async def test_non_jwt_token_skips_cache(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    """A bearer token that is not a valid JWT bypasses the cache."""
    client, _ = mock_client

    await verifier._authenticate_iguazio_v4(_make_request("not-a-jwt"))
    await verifier._authenticate_iguazio_v4(_make_request("not-a-jwt"))

    assert len(verifier._token_cache) == 0
    assert client.verify_request_session.call_count == 2


@pytest.mark.asyncio
async def test_jwt_without_exp_skips_cache(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    """A JWT without an exp claim bypasses the cache."""
    client, _ = mock_client
    token = jwt.encode({"sub": "test-user"}, key="secret", algorithm="HS256")

    await verifier._authenticate_iguazio_v4(_make_request(token))
    await verifier._authenticate_iguazio_v4(_make_request(token))

    assert len(verifier._token_cache) == 0
    assert client.verify_request_session.call_count == 2


@pytest.mark.asyncio
async def test_expired_token_raises_skips_cache(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    """An expired JWT bypassts the cache."""
    client, _ = mock_client
    token = _make_jwt(exp=time.time() - 1)

    await verifier._authenticate_iguazio_v4(_make_request(token))
    await verifier._authenticate_iguazio_v4(_make_request(token))

    assert len(verifier._token_cache) == 0
    assert client.verify_request_session.call_count == 2


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

    assert (
        framework.utils.auth.verifier.AuthVerifier._token_cache_key(token)
        not in verifier._token_cache
    )

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
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.authentication.iguazio.token_cache, "max_size", 2
    )

    tokens = [_make_jwt(exp=time.time() + 3600, sub=f"user_{i}") for i in range(3)]

    for token in tokens:
        await verifier._authenticate_iguazio_v4(_make_request(token))

    assert (
        framework.utils.auth.verifier.AuthVerifier._token_cache_key(tokens[0])
        not in verifier._token_cache
    )
    assert (
        framework.utils.auth.verifier.AuthVerifier._token_cache_key(tokens[1])
        in verifier._token_cache
    )
    assert (
        framework.utils.auth.verifier.AuthVerifier._token_cache_key(tokens[2])
        in verifier._token_cache
    )


@pytest.mark.asyncio
async def test_ttl_expiry(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
):
    client, _ = mock_client
    base_time = 0
    ttl: int = mlrun.mlconf.httpdb.authentication.iguazio.token_cache.ttl_seconds
    token = _make_jwt(exp=base_time + ttl * 10)
    request = _make_request(token)

    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        mock_time.time.return_value = base_time
        await verifier._authenticate_iguazio_v4(request)

    assert client.verify_request_session.call_count == 1
    init_task, init_expires_at = verifier._token_cache[
        framework.utils.auth.verifier.AuthVerifier._token_cache_key(token)
    ]
    assert init_expires_at == base_time + ttl

    # Advance time past TTL; _authenticate_iguazio_v4 should expire the token
    # internally and call the backend again
    refresh_time = init_expires_at + 1

    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        mock_time.time.return_value = refresh_time
        await verifier._authenticate_iguazio_v4(request)

    assert client.verify_request_session.call_count == 2
    refresh_task, refresh_expires_at = verifier._token_cache[
        framework.utils.auth.verifier.AuthVerifier._token_cache_key(token)
    ]
    assert refresh_task is not init_task
    assert refresh_expires_at == refresh_time + ttl


@pytest.mark.asyncio
async def test_ttl_capped_at_token_expiry(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
    monkeypatch: pytest.MonkeyPatch,
):
    """The cache entry expires at the token's own expiry when that is sooner than the TTL."""
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.authentication.iguazio.token_cache, "ttl_seconds", 300
    )
    curr_time = 0
    token_expires_at = 100  # expires before the 300s TTL
    token = _make_jwt(exp=token_expires_at)

    with unittest.mock.patch("framework.utils.auth.verifier.time") as mock_time:
        mock_time.time.return_value = curr_time
        await verifier._authenticate_iguazio_v4(_make_request(token))

    _, cached_expires_at = verifier._token_cache[
        framework.utils.auth.verifier.AuthVerifier._token_cache_key(token)
    ]
    assert cached_expires_at == token_expires_at


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

    assert result1.username == auth_info.username
    assert result2.username == auth_info.username
    mock_instance.verify_request_session.assert_awaited_once()


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
    ttl = mlrun.mlconf.httpdb.authentication.iguazio.token_cache.ttl_seconds
    token = _make_jwt(exp=base_time + ttl * 10)

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
        mock_time.time.return_value = (
            base_time
            + mlrun.mlconf.httpdb.authentication.iguazio.token_cache.ttl_seconds
            + 1
        )
        result = await verifier._authenticate_iguazio_v4(_make_request(token))

    assert result.username == "new-user"

    # Release task_v1 to fail; its done callback must not evict task_v2
    backend_proceed.set()
    with pytest.raises(Exception, match="old task failed"):
        await task_outer
    await asyncio.sleep(0)  # let the done callback run

    assert (
        framework.utils.auth.verifier.AuthVerifier._token_cache_key(token)
        in verifier._token_cache
    ), "task_v2 should still be cached after stale task_v1 callback fires"


@pytest.mark.parametrize(
    "authorization, prefix, expected",
    [
        # Exact match returns the value after the prefix
        ("Basic dXNlcjpwYXNz", "Basic ", "dXNlcjpwYXNz"),
        ("Bearer mytoken", "Bearer ", "mytoken"),
        # Wrong scheme returns None
        ("Bearer mytoken", "Basic ", None),
        ("Basic dXNlcjpwYXNz", "Bearer ", None),
        # Missing header returns None
        (None, "Bearer ", None),
        # Case-insensitive scheme: lowercase
        ("basic dXNlcjpwYXNz", "Basic ", "dXNlcjpwYXNz"),
        ("bearer mytoken", "Bearer ", "mytoken"),
        # Case-insensitive scheme: uppercase
        ("BASIC dXNlcjpwYXNz", "Basic ", "dXNlcjpwYXNz"),
        ("BEARER mytoken", "Bearer ", "mytoken"),
        # Case-insensitive scheme: mixed case
        ("bAsIc dXNlcjpwYXNz", "Basic ", "dXNlcjpwYXNz"),
        ("BeArEr mytoken", "Bearer ", "mytoken"),
    ],
)
def test_parse_auth_header(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    authorization: str | None,
    prefix: str,
    expected: str | None,
):
    headers = _make_headers(authorization)
    assert verifier._parse_auth_header(headers, prefix) == expected


@pytest.mark.parametrize("scheme", ["Basic", "basic", "BASIC", "bAsIc"])
def test_authenticate_basic_case_insensitive_scheme(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    scheme: str,
):
    import base64

    mlrun.mlconf.httpdb.authentication.mode = "basic"
    mlrun.mlconf.httpdb.authentication.basic.username = "user"
    mlrun.mlconf.httpdb.authentication.basic.password = "pass"

    encoded = base64.b64encode(b"user:pass").decode()
    headers = _make_headers(f"{scheme} {encoded}")

    auth_info = verifier._authenticate_basic(headers)
    assert auth_info.username == "user"
    assert auth_info.password == "pass"


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER", "bEaReR"])
def test_authenticate_bearer_case_insensitive_scheme(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    scheme: str,
):
    mlrun.mlconf.httpdb.authentication.mode = "bearer"
    mlrun.mlconf.httpdb.authentication.bearer.token = "secret"

    headers = _make_headers(f"{scheme} secret")

    auth_info = verifier._authenticate_bearer(headers)
    assert auth_info.token == "secret"


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER", "bEaReR"])
async def test_authenticate_iguazio_v4_case_insensitive_scheme_uses_cache(
    verifier: framework.utils.auth.verifier.AuthVerifier,
    mock_client: tuple[unittest.mock.AsyncMock, schemas.AuthInfo],
    scheme: str,
):
    """Any capitalisation of 'Bearer' should be accepted and cached."""
    client, auth_info = mock_client
    token = _make_jwt(exp=time.time() + 3600)

    request = fastapi.Request({"type": "http"})
    request._headers = _make_headers(f"{scheme} {token}")

    result1 = await verifier._authenticate_iguazio_v4(request)
    result2 = await verifier._authenticate_iguazio_v4(request)

    assert result1.username == auth_info.username
    assert result2.username == auth_info.username
    # Both requests must share the single cached backend call
    client.verify_request_session.assert_awaited_once()


def _make_jwt(exp: float, sub: str = "test-user") -> str:
    """Create a minimal JWT with the given expiry timestamp."""
    return jwt.encode({"exp": int(exp), "sub": sub}, key="secret", algorithm="HS256")


def _make_headers(authorization: str | None) -> starlette.datastructures.Headers:
    headers: dict[str, str] = {}
    if authorization is not None:
        headers["Authorization"] = authorization
    return starlette.datastructures.Headers(headers)


def _make_request(token: str | None, scheme: str = "Bearer") -> fastapi.Request:
    headers = _make_headers(None if token is None else f"{scheme} {token}")
    return fastapi.Request({"type": "http", "headers": headers.raw})
