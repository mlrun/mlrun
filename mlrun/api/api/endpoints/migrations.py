import http

import fastapi

import mlrun.api.crud
import mlrun.api.initial_data
import mlrun.api.schemas
import mlrun.api.utils.background_tasks

router = fastapi.APIRouter()


@router.post(
    "/migrations/start",
    responses={
        http.HTTPStatus.OK.value: {},
        http.HTTPStatus.ACCEPTED.value: {"model": mlrun.api.schemas.BackgroundTask},
    },
)
def start_migration(background_tasks: fastapi.BackgroundTasks,):
    if mlrun.mlconf.httpdb.state == mlrun.api.schemas.APIStates.migration_in_progress:
        raise mlrun.errors.MLRunConflictError("Migration already in progress")
    if mlrun.mlconf.httpdb.state != mlrun.api.schemas.APIStates.waiting_for_migrations:
        return fastapi.Response(status_code=http.HTTPStatus.OK.value)
    return mlrun.api.utils.background_tasks.Handler().create_background_task(
        background_tasks,
        mlrun.api.initial_data.init_data,
        perform_migrations_if_needed=True,
    )
