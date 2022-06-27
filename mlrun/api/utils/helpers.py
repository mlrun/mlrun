import asyncio

import mlrun
import mlrun.api.schemas
from mlrun.utils import logger


def ensure_running_on_chief(function):
    """
    The motivation of this function is to catch development bugs in which we are accidentally using functions / flows
    that are supposed to run only in chief instance and by mistake got involved in a worker instance flow.

    Note that there is an option to disable this behavior, its not recommended at all, because it can cause the
    cluster to get out of synchronization.
    """

    def _ensure_running_on_chief():
        if (
            mlrun.mlconf.httpdb.clusterization.role
            != mlrun.api.schemas.ClusterizationRole.chief
        ):
            if (
                mlrun.mlconf.httpdb.clusterization.ensure_function_running_on_chief_mode
                == "enabled"
            ):
                message = (
                    f"{function.__name__} is supposed to run only on chief, re-route."
                )
                raise mlrun.errors.MLRunConflictError(message)
            else:
                logger.warning(
                    f"running {function.__name__} chief function on worker",
                    fail_mode=mlrun.mlconf.httpdb.clusterization.ensure_function_running_on_chief_mode,
                )

    def wrapper(*args, **kwargs):
        _ensure_running_on_chief()
        return function(*args, **kwargs)

    async def async_wrapper(*args, **kwargs):
        _ensure_running_on_chief()
        return await function(*args, **kwargs)

    if asyncio.iscoroutinefunction(function):
        return async_wrapper
    return wrapper
