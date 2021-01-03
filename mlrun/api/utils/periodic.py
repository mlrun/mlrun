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
async def _periodic_function_wrapper(interval: int, function, *args, **kwargs):
    while True:
        try:
            if asyncio.iscoroutinefunction(function):
                await function(*args, **kwargs)
            else:
                await run_in_threadpool(function, *args, **kwargs)
        except Exception:
            logger.warning(
                f"Failed during periodic function execution: {function.__name__}, exc: {traceback.format_exc()}"
            )
        await asyncio.sleep(interval)


def run_function_periodically(
    interval: int, name: str, replace: bool, function, *args, **kwargs
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
        tasks[name].cancel()
        del tasks[name]


def cancel_all_periodic_functions():
    global tasks
    logger.debug("Canceling periodic functions", functions=tasks.keys())
    for task in tasks.values():
        task.cancel()
    tasks = {}
