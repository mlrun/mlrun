import typing
from http import HTTPStatus

from fastapi import APIRouter, Depends, Response, Header, Query
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.utils.singletons.project_member import get_project_member

router = APIRouter()


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' http://localhost:8080/project
@router.post("/projects", response_model=schemas.Project)
def create_project(
    project: schemas.Project, db_session: Session = Depends(deps.get_db_session),
):
    return get_project_member().create_project(db_session, project)


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' -X UPDATE http://localhost:8080/project
@router.put("/projects/{name}", response_model=schemas.Project)
def store_project(
    project: schemas.Project,
    name: str,
    db_session: Session = Depends(deps.get_db_session),
):
    return get_project_member().store_project(db_session, name, project)


@router.patch("/projects/{name}", response_model=schemas.Project)
def patch_project(
    project: dict,
    name: str,
    patch_mode: schemas.PatchMode = Header(
        schemas.PatchMode.replace, alias=schemas.HeaderNames.patch_mode
    ),
    db_session: Session = Depends(deps.get_db_session),
):
    return get_project_member().patch_project(db_session, name, project, patch_mode)


# curl http://localhost:8080/project/<name>
@router.get("/projects/{name}", response_model=schemas.Project)
def get_project(name: str, db_session: Session = Depends(deps.get_db_session)):
    return get_project_member().get_project(db_session, name)


@router.delete("/projects/{name}", status_code=HTTPStatus.NO_CONTENT.value)
def delete_project(
    name: str,
    deletion_strategy: schemas.DeletionStrategy = Header(
        schemas.DeletionStrategy.default(), alias=schemas.HeaderNames.deletion_strategy
    ),
    db_session: Session = Depends(deps.get_db_session),
):
    get_project_member().delete_project(db_session, name, deletion_strategy)
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
