import fastapi
import semver

import mlrun.api.api.deps
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.background_tasks

router = fastapi.APIRouter()


@router.get(
    "/projects/{project}/background-tasks/{name}",
    response_model=mlrun.api.schemas.BackgroundTask,
)
def get_project_background_task(
    project: str,
    name: str,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    # Since there's no not-found option on get_project_background_task - we authorize before getting (unlike other
    # get endpoint)
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.project_background_task,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    return mlrun.api.utils.background_tasks.Handler().get_project_background_task(
        project, name
    )


@router.get(
    "/background-tasks/{name}", response_model=mlrun.api.schemas.BackgroundTask,
)
def get_background_task(
    name: str,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    # Since there's no not-found option on get_background_task - we authorize before getting (unlike other get endpoint)
    # In Iguazio 3.2 the manifest doesn't support the global background task resource - therefore we have to just omit
    # authorization
    # we also skip Iguazio 3.4 for now, until it will add support for it (still in development)
    igz_version = mlrun.mlconf.get_parsed_igz_version()
    if igz_version and igz_version >= semver.VersionInfo.parse("3.5.0-b1"):
        mlrun.api.utils.auth.verifier.AuthVerifier().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.background_task,
            name,
            mlrun.api.schemas.AuthorizationAction.read,
            auth_info,
        )
    return mlrun.api.utils.background_tasks.Handler().get_background_task(name)
