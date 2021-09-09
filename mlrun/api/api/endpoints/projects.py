import http
import typing

import fastapi
import sqlalchemy.orm

import mlrun.api.api.deps
import mlrun.api.schemas
import mlrun.api.utils.clients.opa
from mlrun.api.utils.singletons.project_member import get_project_member

router = fastapi.APIRouter()


@router.post(
    "/projects",
    responses={
        http.HTTPStatus.CREATED.value: {"model": mlrun.api.schemas.Project},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
def create_project(
    project: mlrun.api.schemas.Project,
    response: fastapi.Response,
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_verifier: mlrun.api.api.deps.AuthVerifierDep = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifierDep
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    project, is_running_in_background = get_project_member().create_project(
        db_session,
        project,
        auth_verifier.auth_info.projects_role,
        auth_verifier.auth_info.session,
        wait_for_completion=wait_for_completion,
    )
    if is_running_in_background:
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)
    response.status_code = http.HTTPStatus.CREATED.value
    return project


@router.put(
    "/projects/{name}",
    responses={
        http.HTTPStatus.OK.value: {"model": mlrun.api.schemas.Project},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
def store_project(
    project: mlrun.api.schemas.Project,
    name: str,
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_verifier: mlrun.api.api.deps.AuthVerifierDep = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifierDep
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
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
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)
    return project


@router.patch(
    "/projects/{name}",
    responses={
        http.HTTPStatus.OK.value: {"model": mlrun.api.schemas.Project},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
def patch_project(
    project: dict,
    name: str,
    patch_mode: mlrun.api.schemas.PatchMode = fastapi.Header(
        mlrun.api.schemas.PatchMode.replace,
        alias=mlrun.api.schemas.HeaderNames.patch_mode,
    ),
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_verifier: mlrun.api.api.deps.AuthVerifierDep = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifierDep
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
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
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)
    return project


@router.get("/projects/{name}", response_model=mlrun.api.schemas.Project)
def get_project(
    name: str,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
    auth_verifier: mlrun.api.api.deps.AuthVerifierDep = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifierDep
    ),
):
    project = get_project_member().get_project(
        db_session, name, auth_verifier.auth_info.session
    )
    # skip permission check if it's the leader
    if not _is_request_from_leader(auth_verifier.auth_info.projects_role):
        mlrun.api.utils.clients.opa.Client().query_project_permissions(
            name, mlrun.api.schemas.AuthorizationAction.read, auth_verifier.auth_info,
        )
    return project


@router.delete(
    "/projects/{name}",
    responses={
        http.HTTPStatus.NO_CONTENT.value: {},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
def delete_project(
    name: str,
    deletion_strategy: mlrun.api.schemas.DeletionStrategy = fastapi.Header(
        mlrun.api.schemas.DeletionStrategy.default(),
        alias=mlrun.api.schemas.HeaderNames.deletion_strategy,
    ),
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_verifier: mlrun.api.api.deps.AuthVerifierDep = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifierDep
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
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
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)
    return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)


@router.get("/projects", response_model=mlrun.api.schemas.ProjectsOutput)
def list_projects(
    format_: mlrun.api.schemas.ProjectsFormat = fastapi.Query(
        mlrun.api.schemas.ProjectsFormat.full, alias="format"
    ),
    owner: str = None,
    labels: typing.List[str] = fastapi.Query(None, alias="label"),
    state: mlrun.api.schemas.ProjectState = None,
    auth_verifier: mlrun.api.api.deps.AuthVerifierDep = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifierDep
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    projects_output = get_project_member().list_projects(
        db_session,
        owner,
        mlrun.api.schemas.ProjectsFormat.name_only,
        labels,
        state,
        auth_verifier.auth_info.projects_role,
        auth_verifier.auth_info.session,
    )
    allowed_project_names = projects_output.projects
    # skip permission check if it's the leader
    if not _is_request_from_leader(auth_verifier.auth_info.projects_role):
        allowed_project_names = mlrun.api.utils.clients.opa.Client().filter_projects_by_permissions(
            projects_output.projects, auth_verifier.auth_info,
        )
    return get_project_member().list_projects(
        db_session,
        owner,
        format_,
        labels,
        state,
        auth_verifier.auth_info.projects_role,
        auth_verifier.auth_info.session,
        allowed_project_names,
    )


def _is_request_from_leader(
    projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole],
) -> bool:
    if projects_role and projects_role.value == mlrun.mlconf.httpdb.projects.leader:
        return True
    return False
