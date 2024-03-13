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
import http
import typing

import fastapi
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas
import server.py.services.api.initial_data
import server.py.services.api.utils.background_tasks
import server.py.services.api.utils.clients.chief
from mlrun.utils import logger

router = fastapi.APIRouter()


current_migration_background_task_name = None


@router.post(
    "/operations/migrations",
    responses={
        http.HTTPStatus.OK.value: {},
        http.HTTPStatus.ACCEPTED.value: {"model": mlrun.common.schemas.BackgroundTask},
    },
)
async def trigger_migrations(
    background_tasks: fastapi.BackgroundTasks,
    response: fastapi.Response,
    request: fastapi.Request,
):
    # only chief can execute migrations, redirecting request to chief
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info("Requesting to trigger migrations, re-routing to chief")
        chief_client = server.py.services.api.utils.clients.chief.Client()
        return await chief_client.trigger_migrations(request)

    # we didn't yet decide who should have permissions to such actions, therefore no authorization at the moment
    # note in api.py we do declare to use the authenticate_request dependency - meaning we do have authentication
    global current_migration_background_task_name

    task_callback, background_task, task_name = await run_in_threadpool(
        _get_or_create_migration_background_task,
        current_migration_background_task_name,
    )
    if not task_callback and not background_task:
        # Not waiting for migrations, returning OK
        return fastapi.Response(status_code=http.HTTPStatus.OK.value)

    if not background_task:
        # No task in progress, creating a new one
        background_tasks.add_task(task_callback)
        background_task = server.py.services.api.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task(
            task_name
        )

    response.status_code = http.HTTPStatus.ACCEPTED.value
    current_migration_background_task_name = background_task.metadata.name
    return background_task


def _get_or_create_migration_background_task(
    task_name: str,
) -> tuple[
    typing.Optional[typing.Callable],
    typing.Optional[mlrun.common.schemas.BackgroundTask],
    str,
]:
    if (
        mlrun.mlconf.httpdb.state
        == mlrun.common.schemas.APIStates.migrations_in_progress
    ):
        background_task = server.py.services.api.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task(
            task_name
        )
        return None, background_task, task_name
    elif mlrun.mlconf.httpdb.state == mlrun.common.schemas.APIStates.migrations_failed:
        raise mlrun.errors.MLRunPreconditionFailedError(
            "Migrations were already triggered and failed. Restart the API to retry"
        )
    elif (
        mlrun.mlconf.httpdb.state
        != mlrun.common.schemas.APIStates.waiting_for_migrations
    ):
        return None, None, ""

    logger.info("Starting the migration process")
    (
        task,
        task_name,
    ) = server.py.services.api.utils.background_tasks.InternalBackgroundTasksHandler().create_background_task(
        server.py.services.api.utils.background_tasks.BackgroundTaskKinds.db_migrations,
        None,
        _perform_migration,
    )
    return task, None, task_name


async def _perform_migration():
    # import here to prevent import cycle
    import server.py.services.api.main

    await run_in_threadpool(
        server.py.services.api.initial_data.init_data, perform_migrations_if_needed=True
    )
    await server.py.services.api.main.move_api_to_online()
    mlrun.mlconf.httpdb.state = mlrun.common.schemas.APIStates.online
