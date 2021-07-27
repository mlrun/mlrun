import fastapi

import mlrun.api.api.deps
import mlrun.api.schemas
import mlrun.api.utils.background_tasks

router = fastapi.APIRouter()


@router.get(
    "/projects/{project}/background-tasks/{name}",
    response_model=mlrun.api.schemas.BackgroundTask,
)
def get_background_task(
    project: str,
    name: str,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
):
    return mlrun.api.utils.background_tasks.Handler().get_background_task(
        project, name, auth_verifier.auth_info
    )
