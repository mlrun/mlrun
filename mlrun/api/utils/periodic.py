import asyncio
from typing import List, Callable
from mlrun.utils import logger
from fastapi.concurrency import run_in_threadpool

tasks: List = []


# This module is different from mlrun.db.periodic in that this module's functions aren't supposed to persist
# also this module supports asyncio while the other currently not
# TODO: merge the modules
async def _periodic_function_wrapper(interval, function, *args, **kwargs):
    while True:
        if asyncio.iscoroutinefunction(function):
            await function(*args, **kwargs)
        else:
            await run_in_threadpool(function, *args, **kwargs)
        await asyncio.sleep(interval)


def run_function_periodically(interval, function, *args, **kwargs):
    global tasks
    logger.debug(f'Submitting function to run periodically: {function.__name__}')
    loop = asyncio.get_running_loop()
    task = loop.create_task(_periodic_function_wrapper(interval, function, *args, **kwargs))
    tasks.append(task)


def cancel_periodic_functions():
    logger.debug('Canceling periodic functions')
    global tasks
    for task in tasks:
        task.cancel()
