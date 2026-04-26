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
import pathlib

import pytest

import mlrun
import mlrun.launcher.local
from mlrun.runtimes.local import _run_async_handler

assets_path = pathlib.Path(__file__).parent / "assets"
_HANDLER_FILE = str(assets_path / "async_handlers.py")

parametrize_packagers = pytest.mark.parametrize("packagers_enabled", [False, True])


@pytest.fixture(autouse=True)
def _restore_packagers_enabled():
    original = mlrun.mlconf.packagers.enabled
    yield
    mlrun.mlconf.packagers.enabled = original


@parametrize_packagers
def test_async_handler_completes(packagers_enabled: bool) -> None:
    """Async handler runs to completion and outputs are logged."""
    result = _launch("async_handler", packagers_enabled=packagers_enabled)
    assert result.status.state == "completed"
    assert result.status.results.get("async_result") == 42


@parametrize_packagers
def test_sync_handler_unaffected(packagers_enabled: bool) -> None:
    """Sync handler regression: continues to work correctly after async changes."""
    result = _launch("sync_handler", packagers_enabled=packagers_enabled)
    assert result.status.state == "completed"
    assert result.status.results.get("sync_result") == 99


@parametrize_packagers
def test_async_handler_exception(packagers_enabled: bool, capsys) -> None:
    """Exception inside an async handler is captured and logged to stderr."""
    _launch("async_handler_with_error", packagers_enabled=packagers_enabled)
    captured = capsys.readouterr()
    assert "async error from handler" in captured.out


@parametrize_packagers
@pytest.mark.parametrize(
    "handler_name", ["sync_generator_handler", "async_generator_handler"]
)
def test_generator_raises(packagers_enabled: bool, handler_name: str, capsys) -> None:
    """Generator returns (sync and async) are rejected: error is logged to output."""
    _launch(handler_name, packagers_enabled=packagers_enabled)
    captured = capsys.readouterr()
    assert "generator" in captured.out.lower()


@parametrize_packagers
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


def test_run_async_handler_returns_value() -> None:
    async def coro() -> int:
        await asyncio.sleep(0)
        return 77

    assert _run_async_handler(coro()) == 77


def test_run_async_handler_propagates_exception() -> None:
    async def failing_coro() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run_async_handler(failing_coro())


@parametrize_packagers
@pytest.mark.parametrize(
    "handler_name, result_key, expected, params",
    [
        ("AsyncHandlerClass::run", "class_async_result", 7, None),
        ("AsyncHandlerClass::class_run", "classmethod_async_result", 11, None),
        ("AsyncHandlerClass::static_run", "staticmethod_async_result", 13, None),
        # _init_args are unpacked as kwargs to __init__; multiplier=2 → 7*2=14
        (
            "InitArgsHandlerClass::run",
            "init_args_result",
            14,
            {"_init_args": {"multiplier": 2}},
        ),
        # sync handler: _init_args also work without async
        (
            "SyncInitArgsHandlerClass::run",
            "above_threshold",
            True,
            {"_init_args": {"threshold": 0.9}},
        ),
    ],
)
def test_async_handler_in_class(
    packagers_enabled: bool,
    handler_name: str,
    result_key: str,
    expected: int,
    params: dict | None,
) -> None:
    result = _launch(handler_name, packagers_enabled=packagers_enabled, params=params)
    assert result.status.state == "completed"
    assert result.status.results.get(result_key) == expected


def test_run_async_handler_inside_running_loop() -> None:
    async def coro() -> int:
        await asyncio.sleep(0)
        return 55

    async def call_from_running_loop() -> int:
        # asyncio.get_running_loop() succeeds here, so _run_async_handler
        # will take the ThreadPoolExecutor path.
        return _run_async_handler(coro())

    assert asyncio.run(call_from_running_loop()) == 55


def _launch(
    handler_name: str,
    packagers_enabled: bool = False,
    params: dict | None = None,
) -> mlrun.run.RunObject:
    """Run a handler from the async_handlers asset file via the local launcher."""
    mlrun.mlconf.packagers.enabled = packagers_enabled
    launcher = mlrun.launcher.local.ClientLocalLauncher(local=True)
    runtime = mlrun.code_to_function(
        name="test-async",
        kind="job",
        filename=_HANDLER_FILE,
        handler=handler_name,
    )
    task = mlrun.new_task(params=params) if params else None
    return launcher.launch(runtime, task=task)
