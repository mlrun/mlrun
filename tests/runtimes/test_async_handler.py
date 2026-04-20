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

"""Unit tests for async job handler support (ML-11839).

Tests cover:
- Async handler executes correctly and logs outputs
- Sync handler is unaffected by the async changes
- Exception inside an async handler sets run state to ``error``
- Sync and async generator returns raise ``MLRunRuntimeError``
- Async handler dispatched correctly when a loop is already running (Jupyter path)

Launch-based tests are parametrized over both execution paths:
``packagers_enabled=False`` → ``exec_from_params`` direct path,
``packagers_enabled=True`` → ``mlrun_handler_decorator`` path.
"""

import asyncio
import pathlib

import pytest

import mlrun
import mlrun.launcher.local
import mlrun.runtimes.utils
from mlrun.utils.helpers import _run_async_handler

assets_path = pathlib.Path(__file__).parent / "assets"
_HANDLER_FILE = str(assets_path / "async_handlers.py")

_parametrize_packagers = pytest.mark.parametrize("packagers_enabled", [False, True])


@pytest.fixture(autouse=True)
def _restore_packagers_enabled():
    original = mlrun.mlconf.packagers.enabled
    yield
    mlrun.mlconf.packagers.enabled = original


def _launch(handler_name: str, packagers_enabled: bool = False) -> mlrun.run.RunObject:
    """Run a handler from the async_handlers asset file via the local launcher."""
    mlrun.mlconf.packagers.enabled = packagers_enabled
    launcher = mlrun.launcher.local.ClientLocalLauncher(local=True)
    runtime = mlrun.code_to_function(
        name="test-async",
        kind="job",
        filename=_HANDLER_FILE,
        handler=handler_name,
    )
    return launcher.launch(runtime)


@_parametrize_packagers
def test_async_handler_completes(packagers_enabled: bool) -> None:
    """Async handler runs to completion and outputs are logged."""
    result = _launch("async_handler", packagers_enabled=packagers_enabled)
    assert result.status.state == "completed"
    assert result.status.results.get("async_result") == 42


@_parametrize_packagers
def test_sync_handler_unaffected(packagers_enabled: bool) -> None:
    """Sync handler regression: continues to work correctly after async changes."""
    result = _launch("sync_handler", packagers_enabled=packagers_enabled)
    assert result.status.state == "completed"
    assert result.status.results.get("sync_result") == 99


@_parametrize_packagers
def test_async_handler_exception(packagers_enabled: bool) -> None:
    """Exception inside an async handler propagates: launcher raises RunError."""
    with pytest.raises(mlrun.runtimes.utils.RunError, match="async error from handler"):
        _launch("async_handler_with_error", packagers_enabled=packagers_enabled)


@_parametrize_packagers
@pytest.mark.parametrize(
    "handler_name", ["sync_generator_handler", "async_generator_handler"]
)
def test_generator_raises(packagers_enabled: bool, handler_name: str) -> None:
    """Generator returns (sync and async) must raise RunError wrapping MLRunRuntimeError."""
    with pytest.raises(mlrun.runtimes.utils.RunError, match="(?i)generator"):
        _launch(handler_name, packagers_enabled=packagers_enabled)


@_parametrize_packagers
def test_async_handler_inside_running_loop(packagers_enabled: bool) -> None:
    """Full-stack Jupyter path: async handler completes when _launch is called from within a running event loop.

    Wrapping _launch in a coroutine means asyncio.get_running_loop() succeeds
    inside _run_async_handler, exercising the ThreadPoolExecutor branch.
    """

    async def call_from_running_loop() -> mlrun.run.RunObject:
        return _launch("async_handler", packagers_enabled=packagers_enabled)

    result = asyncio.run(call_from_running_loop())
    assert result.status.state == "completed"
    assert result.status.results.get("async_result") == 42


def test__run_async_handler_returns_value() -> None:
    """Unit test: _run_async_handler drives a coroutine to completion and returns its value."""

    async def coro() -> int:
        await asyncio.sleep(0)
        return 77

    assert _run_async_handler(coro()) == 77


def test__run_async_handler_propagates_exception() -> None:
    """Unit test: _run_async_handler propagates exceptions raised inside the coroutine."""

    async def failing_coro() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run_async_handler(failing_coro())


def test__run_async_handler_inside_running_loop() -> None:
    """Unit test: _run_async_handler uses ThreadPoolExecutor when a loop is already running (Jupyter path).

    Calling _run_async_handler from within an async function (where
    asyncio.get_running_loop() succeeds) exercises the ThreadPoolExecutor branch.
    """

    async def coro() -> int:
        await asyncio.sleep(0)
        return 55

    async def call_from_running_loop() -> int:
        # asyncio.get_running_loop() succeeds here, so _run_async_handler
        # will take the ThreadPoolExecutor path.
        return _run_async_handler(coro())

    assert asyncio.run(call_from_running_loop()) == 55
