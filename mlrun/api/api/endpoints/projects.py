import typing
from http import HTTPStatus

import fastapi
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.utils.singletons.project_member import get_project_member

router = fastapi.APIRouter()


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' http://localhost:8080/project
@router.post(
    "/projects",
    responses={
        HTTPStatus.CREATED.value: {"model": schemas.Project},
        HTTPStatus.ACCEPTED.value: {},
    },
)
def create_project(
    project: schemas.Project,
    response: fastapi.Response,
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_verifier: deps.AuthVerifier = fastapi.Depends(deps.AuthVerifier),
    db_session: Session = fastapi.Depends(deps.get_db_session),
):
    project, is_running_in_background = get_project_member().create_project(
        db_session,
        project,
        auth_verifier.auth_info.projects_role,
        auth_verifier.auth_info.session,
        wait_for_completion=wait_for_completion,
    )
    if is_running_in_background:
        return fastapi.Response(status_code=HTTPStatus.ACCEPTED.value)
    response.status_code = HTTPStatus.CREATED.value
    return project


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' -X UPDATE http://localhost:8080/project
@router.put(
    "/projects/{name}",
    responses={
        HTTPStatus.OK.value: {"model": schemas.Project},
        HTTPStatus.ACCEPTED.value: {},
    },
)
def store_project(
    project: schemas.Project,
    name: str,
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_verifier: deps.AuthVerifier = fastapi.Depends(deps.AuthVerifier),
    db_session: Session = fastapi.Depends(deps.get_db_session),
):
    project, is_running_in_background = get_project_member().store_project(
        db_session,
        name,
        project,
        auth_verifier.auth_info.projects_role,
        auth_verifier.auth_info.session,
        wait_for_completion=wait_for_completion,
    )
    if is_running_in_background:
        return fastapi.Response(status_code=HTTPStatus.ACCEPTED.value)
    return project


@router.patch(
    "/projects/{name}",
    responses={
        HTTPStatus.OK.value: {"model": schemas.Project},
        HTTPStatus.ACCEPTED.value: {},
    },
)
def patch_project(
    project: dict,
    name: str,
    patch_mode: schemas.PatchMode = fastapi.Header(
        schemas.PatchMode.replace, alias=schemas.HeaderNames.patch_mode
    ),
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_verifier: deps.AuthVerifier = fastapi.Depends(deps.AuthVerifier),
    db_session: Session = fastapi.Depends(deps.get_db_session),
):
    project, is_running_in_background = get_project_member().patch_project(
        db_session,
        name,
        project,
        patch_mode,
        auth_verifier.auth_info.projects_role,
        auth_verifier.auth_info.session,
        wait_for_completion=wait_for_completion,
    )
    if is_running_in_background:
        return fastapi.Response(status_code=HTTPStatus.ACCEPTED.value)
    return project


# curl http://localhost:8080/project/<name>
@router.get("/projects/{name}", response_model=schemas.Project)
def get_project(
    name: str,
    db_session: Session = fastapi.Depends(deps.get_db_session),
    auth_verifier: deps.AuthVerifier = fastapi.Depends(deps.AuthVerifier),
):
    return get_project_member().get_project(
        db_session, name, auth_verifier.auth_info.session
    )


@router.delete(
    "/projects/{name}",
    responses={HTTPStatus.NO_CONTENT.value: {}, HTTPStatus.ACCEPTED.value: {}},
)
def delete_project(
    name: str,
    deletion_strategy: schemas.DeletionStrategy = fastapi.Header(
        schemas.DeletionStrategy.default(), alias=schemas.HeaderNames.deletion_strategy
    ),
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_verifier: deps.AuthVerifier = fastapi.Depends(deps.AuthVerifier),
    db_session: Session = fastapi.Depends(deps.get_db_session),
):
    is_running_in_background = get_project_member().delete_project(
        db_session,
        name,
        deletion_strategy,
        auth_verifier.auth_info.projects_role,
        auth_verifier.auth_info,
        wait_for_completion=wait_for_completion,
    )
    if is_running_in_background:
        return fastapi.Response(status_code=HTTPStatus.ACCEPTED.value)
    return fastapi.Response(status_code=HTTPStatus.NO_CONTENT.value)


# curl http://localhost:8080/projects?full=true
@router.get("/projects", response_model=schemas.ProjectsOutput)
def list_projects(
    format_: schemas.ProjectsFormat = fastapi.Query(
        schemas.ProjectsFormat.full, alias="format"
    ),
    owner: str = None,
    labels: typing.List[str] = fastapi.Query(None, alias="label"),
    state: schemas.ProjectState = None,
    auth_verifier: deps.AuthVerifier = fastapi.Depends(deps.AuthVerifier),
    db_session: Session = fastapi.Depends(deps.get_db_session),
):
    return get_project_member().list_projects(
        db_session,
        owner,
        format_,
        labels,
        state,
        auth_verifier.auth_info.projects_role,
        auth_verifier.auth_info.session,
    )
