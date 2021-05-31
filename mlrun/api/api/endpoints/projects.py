import typing
from http import HTTPStatus

from fastapi import APIRouter, Depends, Header, Query, Response
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.utils.singletons.project_member import get_project_member

router = APIRouter()


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
    response: Response,
    projects_role: typing.Optional[schemas.ProjectsRole] = Header(
        None, alias=schemas.HeaderNames.projects_role
    ),
    db_session: Session = Depends(deps.get_db_session),
):
    project, is_running_in_background = get_project_member().create_project(
        db_session, project, projects_role, wait_for_completion=False
    )
    if is_running_in_background:
        return Response(status_code=HTTPStatus.ACCEPTED.value)
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
    projects_role: typing.Optional[schemas.ProjectsRole] = Header(
        None, alias=schemas.HeaderNames.projects_role
    ),
    db_session: Session = Depends(deps.get_db_session),
):
    project, is_running_in_background = get_project_member().store_project(
        db_session, name, project, projects_role, wait_for_completion=False
    )
    if is_running_in_background:
        return Response(status_code=HTTPStatus.ACCEPTED.value)
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
    patch_mode: schemas.PatchMode = Header(
        schemas.PatchMode.replace, alias=schemas.HeaderNames.patch_mode
    ),
    projects_role: typing.Optional[schemas.ProjectsRole] = Header(
        None, alias=schemas.HeaderNames.projects_role
    ),
    db_session: Session = Depends(deps.get_db_session),
):
    project, is_running_in_background = get_project_member().patch_project(
        db_session, name, project, patch_mode, projects_role, wait_for_completion=False
    )
    if is_running_in_background:
        return Response(status_code=HTTPStatus.ACCEPTED.value)
    return project


# curl http://localhost:8080/project/<name>
@router.get("/projects/{name}", response_model=schemas.Project)
def get_project(name: str, db_session: Session = Depends(deps.get_db_session)):
    return get_project_member().get_project(db_session, name)


@router.delete(
    "/projects/{name}",
    responses={HTTPStatus.NO_CONTENT.value: {}, HTTPStatus.ACCEPTED.value: {}},
)
def delete_project(
    name: str,
    deletion_strategy: schemas.DeletionStrategy = Header(
        schemas.DeletionStrategy.default(), alias=schemas.HeaderNames.deletion_strategy
    ),
    projects_role: typing.Optional[schemas.ProjectsRole] = Header(
        None, alias=schemas.HeaderNames.projects_role
    ),
    db_session: Session = Depends(deps.get_db_session),
):
    is_running_in_background = get_project_member().delete_project(
        db_session, name, deletion_strategy, projects_role, wait_for_completion=False
    )
    if is_running_in_background:
        return Response(status_code=HTTPStatus.ACCEPTED.value)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


# curl http://localhost:8080/projects?full=true
@router.get("/projects", response_model=schemas.ProjectsOutput)
def list_projects(
    format_: schemas.Format = Query(schemas.Format.full, alias="format"),
    owner: str = None,
    labels: typing.List[str] = Query(None, alias="label"),
    state: schemas.ProjectState = None,
    db_session: Session = Depends(deps.get_db_session),
):
    return get_project_member().list_projects(db_session, owner, format_, labels, state)
