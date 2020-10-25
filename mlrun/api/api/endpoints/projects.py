from operator import attrgetter
from http import HTTPStatus

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.api.db.sqldb.helpers import to_dict as db2dict
from mlrun.api.utils.singletons.db import get_db
from mlrun import new_project

router = APIRouter()


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' http://localhost:8080/project
@router.post("/project")
def add_project(
    project: schemas.ProjectCreate,
    use_vault=False,
    db_session: Session = Depends(deps.get_db_session),
):
    project_id = get_db().add_project(db_session, project.dict())

    if use_vault:
        proj = new_project(project.name, use_vault=True)
        proj.init_vault()

    return {
        "id": project_id,
        "name": project.name,
    }


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' -X UPDATE http://localhost:8080/project
@router.post("/project/{name}")
def update_project(
    project: schemas.ProjectUpdate,
    name: str,
    use_vault=False,
    db_session: Session = Depends(deps.get_db_session),
):
    if project.name and project.name != name:
        log_and_raise(error=f"Conflict between path proj name {name} and project name {project.name}")

    proj = get_db().get_project(db_session, name)
    if not proj:
        project_id = get_db().add_project(db_session, project.dict())
    else:
        project_id = proj.id
        get_db().update_project(db_session, name, project.dict(exclude_unset=True))

    if use_vault:
        proj = new_project(project.name, use_vault=True)
        proj.init_vault()

    return {
        "id": project_id,
        "name": name,
    }


# curl http://localhost:8080/project/<name>
@router.get("/project/{name}", response_model=schemas.ProjectOut)
def get_project(name: str, db_session: Session = Depends(deps.get_db_session)):
    project = get_db().get_project(db_session, name)
    if not project:
        log_and_raise(error=f"project {name!r} not found")

    project.users = [u.name for u in project.users]

    return {
        "project": project,
    }


@router.delete("/projects/{name}", status_code=HTTPStatus.NO_CONTENT.value)
def delete_project(
    name: str, db_session: Session = Depends(deps.get_db_session),
):
    get_db().delete_project(db_session, name)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


# curl http://localhost:8080/projects?full=true
@router.get("/projects")
def list_projects(
    full: bool = False, db_session: Session = Depends(deps.get_db_session)
):
    fn = db2dict if full else attrgetter("name")
    projects = []
    for p in get_db().list_projects(db_session):
        if isinstance(p, dict):
            if full:
                projects.append(p)
            else:
                projects.append(p.get("name"))
        else:
            projects.append(fn(p))

    return {
        "projects": projects,
    }
