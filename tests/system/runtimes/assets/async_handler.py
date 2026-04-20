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


async def fetch_data(context: mlrun.MLClientCtx) -> int:
    """Async handler that simulates I/O and logs a result."""
    context.logger.info("Async handler started")
    await asyncio.sleep(1)  # simulate async I/O
    result_value = 42
    context.log_result("async_result", result_value)
    context.logger.info("Async handler completed", result=result_value)
    return result_value
