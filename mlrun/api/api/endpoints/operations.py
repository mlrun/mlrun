import http

import fastapi

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
    global current_migration_background_task_name
    if mlrun.mlconf.httpdb.state == mlrun.api.schemas.APIStates.migration_in_progress:
        background_task = mlrun.api.utils.background_tasks.Handler().get_background_task(current_migration_background_task_name)
        response.status_code = http.HTTPStatus.ACCEPTED.value
        return background_task
    if mlrun.mlconf.httpdb.state != mlrun.api.schemas.APIStates.waiting_for_migrations:
        return fastapi.Response(status_code=http.HTTPStatus.OK.value)
    logger.info("Starting the migration process")
    background_task = mlrun.api.utils.background_tasks.Handler().create_background_task(
        background_tasks,
        mlrun.api.initial_data.init_data,
        perform_migrations_if_needed=True,
    )
    current_migration_background_task_name = background_task.metadata.name
    response.status_code = http.HTTPStatus.ACCEPTED.value
    return background_task
