# Copyright 2018 Iguazio
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
#

import asyncio
import typing


async def maybe_coroutine(function_results: typing.Union[typing.Coroutine, typing.Any]):
    """
    If function_results is a coroutine, await it and return the result. Otherwise, return results.
    This is useful for when function callee is not sure if the response should be awaited or not.
    It is required for the function callee to be async. (e.g.: async def).
    """
    if asyncio.iscoroutine(function_results):
        return await function_results
    return function_results
