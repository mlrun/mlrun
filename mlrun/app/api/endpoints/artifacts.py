import asyncio
from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Request, Query
from sqlalchemy.orm import Session

from mlrun.app.api import deps
from mlrun.app.api.utils import json_error
from mlrun.app.main import db
from mlrun.config import config
from mlrun.utils import logger

router = APIRouter()


# curl -d@/path/to/artifcat http://localhost:8080/artifact/p1/7&key=k
@router.post("/artifact/{project}/{uid}/{key:path}")
def store_artifact(
        request: Request,
        project: str,
        uid: str,
        key: str,
        tag: str = "",
        iter: int = 0,
        db_session: Session = Depends(deps.get_db_session)):
    try:
        data = asyncio.run(request.json())
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    logger.debug(data)
    db.store_artifact(db_session, key, data, uid, iter=iter, tag=tag, project=project)
    return {}


# curl http://localhost:8080/artifact/p1/tags
@router.get("/projects/{project}/artifact-tags")
def list_artifact_tags(
        project: str,
        db_session: Session = Depends(deps.get_db_session)):
    tags = db.list_artifact_tags(db_session, project)
    return {
        "project": project,
        "tags": tags,
    }


# curl http://localhost:8080/projects/my-proj/artifact/key?tag=latest
@router.get("/projects/{project}/artifact/{key:path}")
def read_artifact(
        project: str,
        tag: str = "latest",
        iter: int = 0,
        db_session: Session = Depends(deps.get_db_session)):
    data = db.read_artifact(db_session, tag=tag, iter=iter, project=project)
    return {
        "data": data,
    }


# curl -X DELETE http://localhost:8080/artifact/p1&key=k&tag=t
@router.delete("/artifact/{project}/{uid}")
def del_artifact(
        project: str,
        uid: str,
        key: str,
        tag: str = "",
        db_session: Session = Depends(deps.get_db_session)):
    db.del_artifact(db_session, key, tag, project)
    return {}


# curl http://localhost:8080/artifacts?project=p1?label=l1
@router.get("/artifacts")
def list_artifacts(
        project: str = config.default_project,
        name: str = None,
        tag: str = None,
        labels: List[str] = Query([]),
        db_session: Session = Depends(deps.get_db_session)):
    artifacts = db.list_artifacts(db_session, name, project, tag, labels)
    return {
        "artifacts": artifacts,
    }


# curl -X DELETE http://localhost:8080/artifacts?project=p1?label=l1
@router.delete("/artifacts")
def del_artifacts(
        project: str = "",
        name: str = "",
        tag: str = "",
        labels: List[str] = Query([]),
        db_session: Session = Depends(deps.get_db_session)):
    db.del_artifacts(db_session, name, project, tag, labels)
    return {}
