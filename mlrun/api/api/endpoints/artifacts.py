from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Request, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.api.singletons import get_db
from mlrun.config import config
from mlrun.utils import logger

router = APIRouter()


# curl -d@/path/to/artifcat http://localhost:8080/artifact/p1/7&key=k
@router.post("/artifact/{project}/{uid}/{key:path}")
async def store_artifact(
        request: Request,
        project: str,
        uid: str,
        key: str,
        tag: str = "",
        iter: int = 0,
        db_session: Session = Depends(deps.get_db_session)):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    logger.debug(data)
    await run_in_threadpool(get_db().store_artifact, db_session, key, data, uid, iter=iter, tag=tag, project=project)
    return {}


# curl http://localhost:8080/artifact/p1/tags
@router.get("/projects/{project}/artifact-tags")
def list_artifact_tags(
        project: str,
        db_session: Session = Depends(deps.get_db_session)):
    tags = get_db().list_artifact_tags(db_session, project)
    return {
        "project": project,
        "tags": tags,
    }


# curl http://localhost:8080/projects/my-proj/artifact/key?tag=latest
@router.get("/projects/{project}/artifact/{key:path}")
def read_artifact(
        project: str,
        key: str,
        tag: str = "latest",
        iter: int = 0,
        db_session: Session = Depends(deps.get_db_session)):
    data = get_db().read_artifact(db_session, key, tag=tag, iter=iter, project=project)
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
    get_db().del_artifact(db_session, key, tag, project)
    return {}


# curl http://localhost:8080/artifacts?project=p1?label=l1
@router.get("/artifacts")
def list_artifacts(
        project: str = config.default_project,
        name: str = None,
        tag: str = None,
        labels: List[str] = Query([], alias='label'),
        db_session: Session = Depends(deps.get_db_session)):
    artifacts = get_db().list_artifacts(db_session, name, project, tag, labels)
    return {
        "artifacts": artifacts,
    }


# curl -X DELETE http://localhost:8080/artifacts?project=p1?label=l1
@router.delete("/artifacts")
def del_artifacts(
        project: str = "",
        name: str = "",
        tag: str = "",
        labels: List[str] = Query([], alias='label'),
        db_session: Session = Depends(deps.get_db_session)):
    get_db().del_artifacts(db_session, name, project, tag, labels)
    return {}
