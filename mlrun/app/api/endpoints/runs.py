import asyncio
from distutils.util import strtobool
from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Request, Query
from sqlalchemy.orm import Session

from mlrun.app.api import deps
from mlrun.app.api.utils import json_error
from mlrun.app.main import db
from mlrun.utils import logger

router = APIRouter()


# curl -d @/path/to/run.json http://localhost:8080/run/p1/3?commit=yes
@router.post("/run/{project}/{uid}")
def store_run(
        request: Request,
        project: str,
        uid: str,
        iter: int = 0,
        db_session: Session = Depends(deps.get_db_session)):
    try:
        data = asyncio.run(request.json())
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    logger.debug(data)
    db.store_run(db_session, data, uid, project, iter=iter)
    logger.info("store run: {}".format(data))
    return {}


# curl -X PATCH -d @/path/to/run.json http://localhost:8080/run/p1/3?commit=yes
@router.patch("/run/{project}/{uid}")
def update_run(
        request: Request,
        project: str,
        uid: str,
        iter: int = 0,
        db_session: Session = Depends(deps.get_db_session)):
    try:
        data = asyncio.run(request.json())
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    logger.debug(data)
    db.update_run(db_session, data, uid, project, iter=iter)
    logger.info("update run: {}".format(data))
    return {}


# curl http://localhost:8080/run/p1/3
@router.get("/run/{project}/{uid}")
def read_run(
        project: str,
        uid: str,
        iter: int = 0,
        db_session: Session = Depends(deps.get_db_session)):
    data = db.read_run(db_session, uid, project, iter=iter)
    return {
        "data": data,
    }


# curl -X DELETE http://localhost:8080/run/p1/3
@router.delete("/run/{project}/{uid}")
def del_run(
        project: str,
        uid: str,
        tag: int = 0,
        db_session: Session = Depends(deps.get_db_session)):
    db.del_run(db_session, uid, project, iter=iter)
    return {}


# curl http://localhost:8080/runs?project=p1&name=x&label=l1&label=l2&sort=no
@router.get("/runs")
def list_runs(
        project: str = None,
        name: str = None,
        uid: str = None,
        labels: List[str] = Query([]),
        state: str = None,
        last: int = 0,
        sort: str = "on",
        iter: str = "on",
        db_session: Session = Depends(deps.get_db_session)):
    sort = strtobool(sort)
    iter = strtobool(iter)
    runs = db.list_runs(
        db_session,
        name=name,
        uid=uid,
        project=project,
        labels=labels,
        state=state,
        sort=sort,
        last=last,
        iter=iter,
    )
    return {
        "runs": runs,
    }


# curl -X DELETE http://localhost:8080/runs?project=p1&name=x&days_ago=3
@router.delete("/runs")
def del_runs(
        project: str = None,
        name: str = None,
        labels: List[str] = Query([]),
        state: str = None,
        days_ago: int = 0,
        db_session: Session = Depends(deps.get_db_session)):
    db.del_runs(db_session, name, project, labels, state, days_ago)
    return {}
