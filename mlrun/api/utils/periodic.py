# Copyright 2023 Iguazio
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
import traceback
import typing

from fastapi.concurrency import run_in_threadpool

import mlrun.errors
from mlrun.utils import logger

tasks: typing.Dict = {}


# This module is different from mlrun.db.periodic in that this module's functions aren't supposed to persist
# also this module supports asyncio while the other currently not
# TODO: merge the modules
async def _periodic_function_wrapper(
    interval: typing.Union[int, float], function, *args, **kwargs
):
    while True:
        try:
            if asyncio.iscoroutinefunction(function):
                await function(*args, **kwargs)
            else:
                await run_in_threadpool(function, *args, **kwargs)
        except Exception as exc:
            logger.warning(
                "Failed during periodic function execution",
                func_name=function.__name__,
                exc=mlrun.errors.err_to_str(exc),
                tb=traceback.format_exc(),
            )
        await asyncio.sleep(interval)


def run_function_periodically(
    interval: typing.Union[float, int],
    name: str,
    replace: bool,
    function,
    *args,
    **kwargs
):
    global tasks
    logger.debug("Submitting function to run periodically", name=name)
    if name in tasks:
        if not replace:
            message = "Task with that name already exists"
            logger.warning(message, name=name)
            raise mlrun.errors.MLRunInvalidArgumentError(message)
        cancel_periodic_function(name)
    loop = asyncio.get_running_loop()
    task = loop.create_task(
        _periodic_function_wrapper(interval, function, *args, **kwargs)
    )
    tasks[name] = task


def cancel_periodic_function(name: str):
    global tasks
    logger.debug("Canceling periodic function", name=name)
    if name in tasks:
        task = tasks[name]
        # to enable periodic functions to cancel themselves we first remove the task and then cancel it
        del tasks[name]
        task.cancel()


def cancel_all_periodic_functions():
    global tasks
    logger.debug("Canceling periodic functions", functions=tasks.keys())
    for task in tasks.values():
        task.cancel()
    tasks = {}
