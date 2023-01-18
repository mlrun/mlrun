# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
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
        mlrun.api.crud.Runs().store_run,
        db_session,
        data,
        uid,
        iter,
        project,
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
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
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
        mlrun.api.crud.Runs().update_run,
        db_session,
        project,
        uid,
        iter,
        data,
    )
    return {}


@router.get("/run/{project}/{uid}")
async def get_run(
    project: str,
    uid: str,
    iter: int = 0,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    data = await run_in_threadpool(
        mlrun.api.crud.Runs().get_run, db_session, uid, iter, project
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
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
async def delete_run(
    project: str,
    uid: str,
    iter: int = 0,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.run,
        project,
        uid,
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_info,
    )
    await run_in_threadpool(
        mlrun.api.crud.Runs().delete_run,
        db_session,
        uid,
        iter,
        project,
    )
    return {}


@router.get("/runs")
async def list_runs(
    project: str = None,
    name: str = None,
    uid: List[str] = Query([]),
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
    max_partitions: int = Query(0, alias="max-partitions", ge=0),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    if project != "*":
        await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
            project,
            mlrun.api.schemas.AuthorizationAction.read,
            auth_info,
        )
    runs = await run_in_threadpool(
        mlrun.api.crud.Runs().list_runs,
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
        max_partitions,
    )
    filtered_runs = await mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
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
async def delete_runs(
    project: str = None,
    name: str = None,
    labels: List[str] = Query([], alias="label"),
    state: str = None,
    days_ago: int = None,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    if not project or project != "*":
        # Currently we don't differentiate between runs permissions inside a project.
        # Meaning there is no reason at the moment to query the permission for each run under the project
        # TODO check for every run when we will manage permission per run inside a project
        await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.run,
            project or mlrun.mlconf.default_project,
            "",
            mlrun.api.schemas.AuthorizationAction.delete,
            auth_info,
        )
    else:
        start_time_from = None
        if days_ago:
            start_time_from = datetime.datetime.now(
                datetime.timezone.utc
            ) - datetime.timedelta(days=days_ago)
        runs = await run_in_threadpool(
            mlrun.api.crud.Runs().list_runs,
            db_session,
            name,
            project=project,
            labels=labels,
            states=[state] if state is not None else None,
            start_time_from=start_time_from,
        )
        projects = set(
            run.get("metadata", {}).get("project", mlrun.mlconf.default_project)
            for run in runs
        )
        for run_project in projects:
            # currently we fail if the user doesn't has permissions to delete runs to one of the projects in the system
            # TODO Delete only runs from projects that user has permissions to
            await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
                mlrun.api.schemas.AuthorizationResourceTypes.run,
                run_project,
                "",
                mlrun.api.schemas.AuthorizationAction.delete,
                auth_info,
            )

    await run_in_threadpool(
        mlrun.api.crud.Runs().delete_runs,
        db_session,
        name,
        project,
        labels,
        state,
        days_ago,
    )
    return {}
