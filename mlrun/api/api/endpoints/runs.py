from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.crud
from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
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
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.info("Storing run", data=data)
    await run_in_threadpool(
        mlrun.api.crud.Runs().store_run,
        db_session,
        data,
        uid,
        iter,
        project,
        auth_verifier.auth_info,
    )
    return {}


# curl -X PATCH -d @/path/to/run.json http://localhost:8080/run/p1/3?commit=yes
@router.patch("/run/{project}/{uid}")
async def update_run(
    request: Request,
    project: str,
    uid: str,
    iter: int = 0,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    await run_in_threadpool(
        mlrun.api.crud.Runs().update_run,
        db_session,
        project,
        uid,
        iter,
        data,
        auth_verifier.auth_info,
    )
    return {}


# curl http://localhost:8080/run/p1/3
@router.get("/run/{project}/{uid}")
def get_run(
    project: str,
    uid: str,
    iter: int = 0,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    data = mlrun.api.crud.Runs().get_run(
        db_session, uid, iter, project, auth_verifier.auth_info
    )
    return {
        "data": data,
    }


# curl -X DELETE http://localhost:8080/run/p1/3
@router.delete("/run/{project}/{uid}")
def delete_run(
    project: str,
    uid: str,
    iter: int = 0,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.crud.Runs().delete_run(
        db_session, uid, iter, project, auth_verifier.auth_info
    )
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
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    runs = mlrun.api.crud.Runs().list_runs(
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
        auth_info=auth_verifier.auth_info,
    )
    return {
        "runs": runs,
    }


# curl -X DELETE http://localhost:8080/runs?project=p1&name=x&days_ago=3
@router.delete("/runs")
def delete_runs(
    project: str = None,
    name: str = None,
    labels: List[str] = Query([], alias="label"),
    state: str = None,
    days_ago: int = 0,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.crud.Runs().delete_runs(
        db_session, name, project, labels, state, days_ago, auth_verifier.auth_info
    )
    return {}
