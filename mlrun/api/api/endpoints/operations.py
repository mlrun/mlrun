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
import http
import typing

import fastapi
from fastapi.concurrency import run_in_threadpool

import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.initial_data
import mlrun.api.schemas
import mlrun.api.utils.background_tasks
import mlrun.api.utils.clients.chief
from mlrun.utils import logger

router = fastapi.APIRouter()


current_migration_background_task_name = None


@router.post(
    "/operations/migrations",
    responses={
        http.HTTPStatus.OK.value: {},
        http.HTTPStatus.ACCEPTED.value: {"model": mlrun.api.schemas.BackgroundTask},
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
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        logger.info("Requesting to trigger migrations, re-routing to chief")
        chief_client = mlrun.api.utils.clients.chief.Client()
        return await chief_client.trigger_migrations(request)

    # we didn't yet decide who should have permissions to such actions, therefore no authorization at the moment
    # note in api.py we do declare to use the authenticate_request dependency - meaning we do have authentication
    global current_migration_background_task_name

    background_task = await run_in_threadpool(
        _get_or_create_migration_background_task,
        current_migration_background_task_name,
        background_tasks,
    )
    if not background_task:
        return fastapi.Response(status_code=http.HTTPStatus.OK.value)

    response.status_code = http.HTTPStatus.ACCEPTED.value
    current_migration_background_task_name = background_task.metadata.name
    return background_task


def _get_or_create_migration_background_task(
    task_name: str, background_tasks
) -> typing.Optional[mlrun.api.schemas.BackgroundTask]:
    if mlrun.mlconf.httpdb.state == mlrun.api.schemas.APIStates.migrations_in_progress:
        background_task = mlrun.api.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task(
            task_name
        )
        return background_task
    elif mlrun.mlconf.httpdb.state == mlrun.api.schemas.APIStates.migrations_failed:
        raise mlrun.errors.MLRunPreconditionFailedError(
            "Migrations were already triggered and failed. Restart the API to retry"
        )
    elif (
        mlrun.mlconf.httpdb.state != mlrun.api.schemas.APIStates.waiting_for_migrations
    ):
        return None

    logger.info("Starting the migration process")
    return mlrun.api.utils.background_tasks.InternalBackgroundTasksHandler().create_background_task(
        background_tasks,
        _perform_migration,
    )


async def _perform_migration():
    # import here to prevent import cycle
    import mlrun.api.main

    await run_in_threadpool(
        mlrun.api.initial_data.init_data, perform_migrations_if_needed=True
    )
    await mlrun.api.main.move_api_to_online()
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.online
