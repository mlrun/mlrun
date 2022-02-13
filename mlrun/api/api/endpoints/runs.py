import datetime
from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.project_member
from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.utils import logger
from mlrun.utils.helpers import datetime_from_iso

router = APIRouter()


@router.post("/run/{project}/{uid}")
async def store_run(
    request: Request,
    project: str,
    uid: str,
    iter: int = 0,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await run_in_threadpool(
        mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions,
        mlrun.api.schemas.AuthorizationResourceTypes.run,
        project,
        uid,
        mlrun.api.schemas.AuthorizationAction.store,
        auth_info,
    )
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.info("Storing run", data=data)
    await run_in_threadpool(
        mlrun.api.crud.Runs().store_run, db_session, data, uid, iter, project,
    )
    return {}


@router.patch("/run/{project}/{uid}")
async def update_run(
    request: Request,
    project: str,
    uid: str,
    iter: int = 0,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions,
        mlrun.api.schemas.AuthorizationResourceTypes.run,
        project,
        uid,
        mlrun.api.schemas.AuthorizationAction.update,
        auth_info,
    )
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    await run_in_threadpool(
        mlrun.api.crud.Runs().update_run, db_session, project, uid, iter, data,
    )
    return {}


@router.get("/run/{project}/{uid}")
def get_run(
    project: str,
    uid: str,
    iter: int = 0,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    data = mlrun.api.crud.Runs().get_run(db_session, uid, iter, project)
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.run,
        project,
        uid,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    return {
        "data": data,
    }


@router.delete("/run/{project}/{uid}")
def delete_run(
    project: str,
    uid: str,
    iter: int = 0,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.run,
        project,
        uid,
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_info,
    )
    mlrun.api.crud.Runs().delete_run(
        db_session, uid, iter, project,
    )
    return {}


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
    partition_by: mlrun.api.schemas.RunPartitionByField = Query(
        None, alias="partition-by"
    ),
    rows_per_partition: int = Query(1, alias="rows-per-partition", gt=0),
    partition_sort_by: mlrun.api.schemas.SortField = Query(
        None, alias="partition-sort-by"
    ),
    partition_order: mlrun.api.schemas.OrderType = Query(
        mlrun.api.schemas.OrderType.desc, alias="partition-order"
    ),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    if project != "*":
        mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
            project, mlrun.api.schemas.AuthorizationAction.read, auth_info,
        )
    runs = mlrun.api.crud.Runs().list_runs(
        db_session,
        name,
        uid,
        project,
        labels,
        [state] if state is not None else None,
        sort,
        last,
        iter,
        datetime_from_iso(start_time_from),
        datetime_from_iso(start_time_to),
        datetime_from_iso(last_update_time_from),
        datetime_from_iso(last_update_time_to),
        partition_by,
        rows_per_partition,
        partition_sort_by,
        partition_order,
    )
    filtered_runs = mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.run,
        runs,
        lambda run: (
            run.get("metadata", {}).get("project", mlrun.mlconf.default_project),
            run.get("metadata", {}).get("uid"),
        ),
        auth_info,
    )
    return {
        "runs": filtered_runs,
    }


@router.delete("/runs")
def delete_runs(
    project: str = None,
    name: str = None,
    labels: List[str] = Query([], alias="label"),
    state: str = None,
    days_ago: int = 0,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    start_time_from = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=days_ago
    )
    runs = mlrun.api.crud.Runs().list_runs(
        db_session,
        name,
        project=project,
        labels=labels,
        states=[state] if state is not None else None,
        start_time_from=start_time_from,
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resources_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.run,
        runs,
        lambda run: (
            run.get("metadata", {}).get("project", mlrun.mlconf.default_project),
            run.get("metadata", {}).get("uid"),
        ),
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_info,
    )
    mlrun.api.crud.Runs().delete_runs(
        db_session, name, project, labels, state, days_ago,
    )
    return {}
