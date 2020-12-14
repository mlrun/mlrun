import fastapi

import mlrun.api.schemas
import mlrun.api.utils.background_tasks

router = fastapi.APIRouter()


@router.get(
    "/projects/{project}/background-tasks/{name}",
    response_model=mlrun.api.schemas.BackgroundTask,
)
def get_background_task(
    project: str, name: str,
):
    return mlrun.api.utils.background_tasks.Handler().get_background_task(project, name)
