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
import enum

import mlrun
import mlrun.api.schemas
from mlrun.utils import logger


# TODO: From python 3.11 StrEnum is built-in and this will not be needed
class StrEnum(str, enum.Enum):
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


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
