from http import HTTPStatus

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.utils.singletons.db import get_db

router = APIRouter()


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' http://localhost:8080/project
# TODO: remove the wrong route (/project) before 0.6.0 is out after UI change to new route
@router.post("/project")
@router.post("/projects", response_model=schemas.ProjectOutput)
def create_project(
    project: schemas.ProjectCreate, db_session: Session = Depends(deps.get_db_session)
):
    get_db().create_project(db_session, project)
    return get_db().get_project(db_session, project.name)


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' -X UPDATE http://localhost:8080/project
# TODO: remove the wrong route (/project/{name}) before 0.6.0 is out after UI change to new route
@router.post("/project/{name}")
@router.post("/projects/{name}", response_model=schemas.ProjectOutput)
def update_project(
    project: schemas.ProjectUpdate,
    name: str,
    db_session: Session = Depends(deps.get_db_session),
):
    get_db().update_project(db_session, name, project)
    return get_db().get_project(db_session, name)


# curl http://localhost:8080/project/<name>
# TODO: remove the wrong route (/project/{name}) before 0.6.0 is out after UI change to new route
@router.get("/project/{name}", response_model=schemas.ProjectOutput)
@router.get("/projects/{name}", response_model=schemas.ProjectOutput)
def get_project(name: str, db_session: Session = Depends(deps.get_db_session)):
    return get_db().get_project(db_session, name)


@router.delete("/projects/{name}", status_code=HTTPStatus.NO_CONTENT.value)
def delete_project(
    name: str, db_session: Session = Depends(deps.get_db_session),
):
    get_db().delete_project(db_session, name)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


# curl http://localhost:8080/projects?full=true
@router.get("/projects", response_model=schemas.ProjectsOutput)
def list_projects(
    full: bool = False,
    owner: str = None,
    db_session: Session = Depends(deps.get_db_session),
):
    return get_db().list_projects(db_session, owner, full)
