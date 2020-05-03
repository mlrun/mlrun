from distutils.util import strtobool
from operator import attrgetter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.api.db.sqldb.helpers import to_dict as db2dict
from mlrun.api.singletons import get_db

router = APIRouter()


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' http://localhost:8080/project
@router.post("/project")
def add_project(
        project: schemas.ProjectCreate,
        db_session: Session = Depends(deps.get_db_session)):
    project_id = get_db().add_project(db_session, project.dict())
    return {
        "id": project_id,
        "name": project.name,
    }


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' -X UPDATE http://localhost:8080/project
@router.post("/project/{name}")
def update_project(
        project: schemas.ProjectUpdate,
        name: str,
        db_session: Session = Depends(deps.get_db_session)):
    get_db().update_project(db_session, name, project.dict(exclude_unset=True))
    return {}


# curl http://localhost:8080/project/<name>
@router.get("/project/{name}", response_model=schemas.ProjectOut)
def get_project(
        name: str,
        db_session: Session = Depends(deps.get_db_session)):
    project = get_db().get_project(db_session, name)
    if not project:
        log_and_raise(error=f"project {name!r} not found")

    project.users = [u.name for u in project.users]

    return {
        "project": project,
    }


# curl http://localhost:8080/projects?full=true
@router.get("/projects")
def list_projects(
        full: str = "no",
        db_session: Session = Depends(deps.get_db_session)):
    full = strtobool(full)
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
