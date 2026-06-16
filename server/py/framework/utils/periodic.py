# Copyright 2024 Iguazio
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
import traceback
import typing
import uuid

from fastapi.concurrency import run_in_threadpool

import mlrun.errors
from mlrun.utils import context_id_var, logger

tasks: dict = {}


def run_function_periodically(
    interval: typing.Union[float, int],
    name: str,
    replace: bool,
    function: typing.Callable,
    *args,
    **kwargs,
):
    """
    Schedule ``function`` to run repeatedly in a background asyncio task.

    NOTE: this is NOT a cron scheduler. ``interval`` is the wait time *between*
    runs, applied AFTER each execution finishes -- it is not a fixed wall-clock
    firing period. The loop is: run the function to completion, then
    ``sleep(interval)``, then run again. Consequently the effective period is
    ``execution_time + interval``, a slow execution pushes out the next start by
    exactly that much, executions never overlap, and -- unlike cron -- runs are
    never queued or "caught up" to hit a fixed cadence.

    Exceptions raised by ``function`` are logged and swallowed; the loop keeps
    going (and still waits ``interval`` before the next run).

    :param interval: Seconds to wait between the END of one run and the START of
        the next (not a fixed firing period -- see note above).
    :param name:     Unique task name; used to cancel or replace the task.
    :param replace:  If True, cancel and replace an existing task with the same
        name; if False and the name exists, raises MLRunInvalidArgumentError.
    :param function: Callable to run periodically. May be sync or async; sync
        functions run in a threadpool so they do not block the event loop.
    :param args:     Positional arguments forwarded to ``function`` each run.
    :param kwargs:   Keyword arguments forwarded to ``function`` each run.
    """
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


async def _periodic_function_wrapper(
    interval: typing.Union[int, float], function, *args, **kwargs
):
    context_id_var.set(str(uuid.uuid4()))
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
        # `interval` is the gap BETWEEN runs: we sleep only after the function
        # has finished (or failed), so the next run starts `interval` seconds
        # later. This makes the schedule drift with execution time rather than
        # firing on a fixed cron-like cadence -- see run_function_periodically.
        await asyncio.sleep(interval)
