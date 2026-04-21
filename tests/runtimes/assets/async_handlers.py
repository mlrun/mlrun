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


async def async_handler(context):
    await asyncio.sleep(0)
    result = 42
    context.log_result("async_result", result)
    return result


async def async_handler_with_error(context):
    """Async handler that raises an exception after yielding."""
    await asyncio.sleep(0)
    raise ValueError("async error from handler")


def sync_handler(context):
    """Plain sync handler — must continue to work without modification."""
    result = 99
    context.log_result("sync_result", result)
    return result


def sync_generator_handler(context):
    """Sync generator — not a valid MLRun job handler return type."""
    yield 1
    yield 2


async def async_generator_handler(context):
    """Async generator — not a valid MLRun job handler return type."""
    yield 1
    yield 2
