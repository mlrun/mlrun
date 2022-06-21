import asyncio

import mlrun
import mlrun.api.schemas
from mlrun.utils import logger


def run_only_on_chief(function):
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    async def async_wrapper(*args, **kwargs):
        return await function(*args, **kwargs)

    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        if mlrun.mlconf.fail_on_running_chief_functions_in_worker_mode == "enabled":
            message = f"{function.__name__} is supposed to run only on chief, re-route."
            raise mlrun.errors.MLRunIncompatibleVersionError(message)
        else:
            logger.warning(
                f"running {function.__name__} chief function on worker",
                fail_mode=mlrun.mlconf.fail_on_running_chief_functions_in_worker_mode,
            )

    if asyncio.iscoroutinefunction(function):
        return async_wrapper
    return wrapper
