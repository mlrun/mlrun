from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Request, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.api.utils.singletons.db import get_db
from mlrun.utils import logger
from mlrun.utils.helpers import datetime_from_iso

router = APIRouter()


# curl -d @/path/to/run.json http://localhost:8080/run/p1/3?commit=yes
@router.post("/run/{project}/{uid}")
async def store_run(
    request: Request,
    project: str,
    uid: str,
    iter: int = 0,
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.info("Storing run", data=data)
    await run_in_threadpool(
        get_db().store_run, db_session, data, uid, project, iter=iter
    )
    return {}


# curl -X PATCH -d @/path/to/run.json http://localhost:8080/run/p1/3?commit=yes
@router.patch("/run/{project}/{uid}")
async def update_run(
    request: Request,
    project: str,
    uid: str,
    iter: int = 0,
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.info("Updating run", data=data)
    await run_in_threadpool(
        get_db().update_run, db_session, data, uid, project, iter=iter
    )
    return {}


# curl http://localhost:8080/run/p1/3
@router.get("/run/{project}/{uid}")
def read_run(
    project: str,
    uid: str,
    iter: int = 0,
    db_session: Session = Depends(deps.get_db_session),
):
    data = get_db().read_run(db_session, uid, project, iter=iter)
    return {
        "data": data,
    }


# curl -X DELETE http://localhost:8080/run/p1/3
@router.delete("/run/{project}/{uid}")
def del_run(
    project: str,
    uid: str,
    iter: int = 0,
    db_session: Session = Depends(deps.get_db_session),
):
    get_db().del_run(db_session, uid, project, iter=iter)
    return {}


# curl http://localhost:8080/runs?project=p1&name=x&label=l1&label=l2&sort=no
@router.get("/runs")
def list_runs(
    project: str = None,
    name: str = None,
    uid: str = None,
    labels: List[str] = Query([], alias="label"),
    state: str = None,
    last: int = 0,
    sort: bool = True,
    iter: bool = True,
    start_time_from: str = None,
    start_time_to: str = None,
    last_update_time_from: str = None,
    last_update_time_to: str = None,
    db_session: Session = Depends(deps.get_db_session),
):
    runs = get_db().list_runs(
        db_session,
        name=name,
        uid=uid,
        project=project,
        labels=labels,
        state=state,
        sort=sort,
        last=last,
        iter=iter,
        start_time_from=datetime_from_iso(start_time_from),
        start_time_to=datetime_from_iso(start_time_to),
        last_update_time_from=datetime_from_iso(last_update_time_from),
        last_update_time_to=datetime_from_iso(last_update_time_to),
    )
    return {
        "runs": runs,
    }


# curl -X DELETE http://localhost:8080/runs?project=p1&name=x&days_ago=3
@router.delete("/runs")
def del_runs(
    project: str = None,
    name: str = None,
    labels: List[str] = Query([], alias="label"),
    state: str = None,
    days_ago: int = 0,
    db_session: Session = Depends(deps.get_db_session),
):
    get_db().del_runs(db_session, name, project, labels, state, days_ago)
    return {}
