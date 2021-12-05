import http

import fastapi
import fastapi.concurrency

import mlrun.api.crud
import mlrun.api.initial_data
import mlrun.api.schemas
import mlrun.api.utils.background_tasks
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
def start_migration(
    background_tasks: fastapi.BackgroundTasks, response: fastapi.Response,
):
    # we didn't yet decide who should have permissions to such actions, therefore no authorization at the moment
    # note in api.py we do declare to use the authenticate_request dependency - meaning we do have authentication
    global current_migration_background_task_name
    if mlrun.mlconf.httpdb.state == mlrun.api.schemas.APIStates.migrations_in_progress:
        background_task = mlrun.api.utils.background_tasks.Handler().get_background_task(
            current_migration_background_task_name
        )
        response.status_code = http.HTTPStatus.ACCEPTED.value
        return background_task
    elif mlrun.mlconf.httpdb.state == mlrun.api.schemas.APIStates.migrations_failed:
        raise mlrun.errors.MLRunPreconditionFailedError(
            "Migrations were already triggered and failed. Restart the API to retry"
        )
    elif (
        mlrun.mlconf.httpdb.state != mlrun.api.schemas.APIStates.waiting_for_migrations
    ):
        return fastapi.Response(status_code=http.HTTPStatus.OK.value)
    logger.info("Starting the migration process")
    background_task = mlrun.api.utils.background_tasks.Handler().create_background_task(
        background_tasks, _perform_migration,
    )
    current_migration_background_task_name = background_task.metadata.name
    response.status_code = http.HTTPStatus.ACCEPTED.value
    return background_task


async def _perform_migration():
    # import here to prevent import cycle
    import mlrun.api.main

    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.initial_data.init_data, perform_migrations_if_needed=True
    )
    await mlrun.api.main.move_api_to_online()
    mlrun.mlconf.httpdb.state = mlrun.api.schemas.APIStates.online
