from distutils.util import strtobool
from operator import attrgetter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from mlrun.app import schemas
from mlrun.app.api import deps
from mlrun.app.api.utils import json_error
from mlrun.app.db.sqldb.helpers import to_dict as db2dict
from mlrun.app.main import db

router = APIRouter()


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' http://localhost:8080/project
@router.post("/project")
def add_project(
        project: schemas.ProjectCreate,
        db_session: Session = Depends(deps.get_db_session)):
    project_id = db.add_project(db_session, project.dict())
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
    db.update_project(db_session, name, project.dict())
    return {}


# curl http://localhost:8080/project/<name>
@router.get("/project/{name}", response_model=schemas.Project)
def get_project(
        name: str,
        db_session: Session = Depends(deps.get_db_session)):
    project = db.get_project(db_session, name)
    if not project:
        return json_error(error=f'project {name!r} not found')

    project.users = [u.name for u in project.users]

    return project


# curl http://localhost:8080/projects?full=true
@router.get("/projects")
def list_projects(
        full: str = "on",
        db_session: Session = Depends(deps.get_db_session)):
    full = strtobool(full)
    fn = db2dict if full else attrgetter("name")
    projects = []
    for p in db.list_projects(db_session):
        if isinstance(p, dict):
            if full:
                projects.append(p)
            else:
                projects.append(p.get('name'))
        else:
            projects.append(fn(p))

    return {
        "projects": projects,
    }
