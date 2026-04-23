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

import mlrun


async def async_handler(context: mlrun.MLClientCtx) -> int:
    await asyncio.sleep(0)
    result = 42
    context.log_result("async_result", result)
    return result


async def async_handler_with_error(context: mlrun.MLClientCtx) -> None:
    """Async handler that raises an exception after yielding."""
    await asyncio.sleep(0)
    raise ValueError("async error from handler")


def sync_handler(context: mlrun.MLClientCtx) -> int:
    """Plain sync handler — must continue to work without modification."""
    result = 99
    context.log_result("sync_result", result)
    return result


def sync_generator_handler(context: mlrun.MLClientCtx):
    """Sync generator — not a valid MLRun job handler return type."""
    yield 1
    yield 2


async def async_generator_handler(context: mlrun.MLClientCtx):
    """Async generator — not a valid MLRun job handler return type."""
    yield 1
    yield 2


class InitArgsHandlerClass:
    def __init__(self, context: mlrun.MLClientCtx, multiplier: int = 1) -> None:
        self.multiplier = multiplier
        context.logger.info("Logging in the constructor")

    async def run(self, context: mlrun.MLClientCtx) -> int:
        await asyncio.sleep(0)
        result = 7 * self.multiplier
        context.log_result("init_args_result", result)
        return result


class SyncInitArgsHandlerClass:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def run(self, context: mlrun.MLClientCtx) -> None:
        context.log_result("above_threshold", self.threshold > 0.3)


class AsyncHandlerClass:
    async def run(self, context: mlrun.MLClientCtx) -> int:
        await asyncio.sleep(0)
        result = 7
        context.log_result("class_async_result", result)
        return result

    @classmethod
    async def class_run(cls, context: mlrun.MLClientCtx) -> int:
        await asyncio.sleep(0)
        result = 11
        context.log_result("classmethod_async_result", result)
        return result

    @staticmethod
    async def static_run(context: mlrun.MLClientCtx) -> int:
        await asyncio.sleep(0)
        result = 13
        context.log_result("staticmethod_async_result", result)
        return result
