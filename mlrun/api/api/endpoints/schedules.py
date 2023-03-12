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
from http import HTTPStatus

import fastapi
from fastapi import APIRouter, Depends, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.api.utils
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.clients.chief
import mlrun.api.utils.singletons.project_member
from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.utils.singletons.scheduler import get_scheduler
from mlrun.utils import logger

router = APIRouter()


@router.post("/projects/{project}/schedules")
async def create_schedule(
    project: str,
    schedule: schemas.ScheduleInput,
    request: fastapi.Request,
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
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        project,
        schedule.name,
        mlrun.api.schemas.AuthorizationAction.create,
        auth_info,
    )
    # to reduce redundant load on the chief, we re-route the request only if the user has permissions
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to create schedule, re-routing to chief",
            project=project,
            schedule=schedule.dict(),
        )
        chief_client = mlrun.api.utils.clients.chief.Client()
        return await chief_client.create_schedule(
            project=project,
            request=request,
            json=schedule.dict(),
        )

    if schedule.credentials.access_key:
        auth_info.access_key = schedule.credentials.access_key
    await run_in_threadpool(
        get_scheduler().create_schedule,
        db_session,
        auth_info,
        project,
        schedule.name,
        schedule.kind,
        schedule.scheduled_object,
        schedule.cron_trigger,
        schedule.labels,
        schedule.concurrency_limit,
    )
    return Response(status_code=HTTPStatus.CREATED.value)


@router.put("/projects/{project}/schedules/{name}")
async def update_schedule(
    project: str,
    name: str,
    schedule: schemas.ScheduleUpdate,
    request: fastapi.Request,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.update,
        auth_info,
    )
    # to reduce redundant load on the chief, we re-route the request only if the user has permissions
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to update schedule, re-routing to chief",
            project=project,
            name=name,
            schedule=schedule.dict(),
        )
        chief_client = mlrun.api.utils.clients.chief.Client()
        return await chief_client.update_schedule(
            project=project,
            name=name,
            request=request,
            json=schedule.dict(),
        )

    if schedule.credentials.access_key:
        auth_info.access_key = schedule.credentials.access_key
    await run_in_threadpool(
        get_scheduler().update_schedule,
        db_session,
        auth_info,
        project,
        name,
        schedule.scheduled_object,
        schedule.cron_trigger,
        labels=schedule.labels,
    )
    return Response(status_code=HTTPStatus.OK.value)


@router.get("/projects/{project}/schedules", response_model=schemas.SchedulesOutput)
async def list_schedules(
    project: str,
    name: str = None,
    labels: str = None,
    kind: schemas.ScheduleKinds = None,
    include_last_run: bool = False,
    include_credentials: bool = fastapi.Query(False, alias="include-credentials"),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    schedules = await run_in_threadpool(
        get_scheduler().list_schedules,
        db_session,
        project,
        name,
        kind,
        labels,
        include_last_run,
        include_credentials,
    )
    filtered_schedules = await mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        schedules.schedules,
        lambda schedule: (
            schedule.project,
            schedule.name,
        ),
        auth_info,
    )
    schedules.schedules = filtered_schedules
    return schedules


@router.get(
    "/projects/{project}/schedules/{name}", response_model=schemas.ScheduleOutput
)
async def get_schedule(
    project: str,
    name: str,
    include_last_run: bool = False,
    include_credentials: bool = fastapi.Query(False, alias="include-credentials"),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    schedule = await run_in_threadpool(
        get_scheduler().get_schedule,
        db_session,
        project,
        name,
        include_last_run,
        include_credentials,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    return schedule


@router.post("/projects/{project}/schedules/{name}/invoke")
async def invoke_schedule(
    project: str,
    name: str,
    request: fastapi.Request,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.update,
        auth_info,
    )
    # to reduce redundant load on the chief, we re-route the request only if the user has permissions
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to invoke schedule, re-routing to chief",
            project=project,
            name=name,
        )
        chief_client = mlrun.api.utils.clients.chief.Client()
        return await chief_client.invoke_schedule(
            project=project, name=name, request=request
        )

    return await get_scheduler().invoke_schedule(db_session, auth_info, project, name)


@router.delete(
    "/projects/{project}/schedules/{name}", status_code=HTTPStatus.NO_CONTENT.value
)
async def delete_schedule(
    project: str,
    name: str,
    request: fastapi.Request,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_info,
    )
    # to reduce redundant load on the chief, we re-route the request only if the user has permissions
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to delete schedule, re-routing to chief",
            project=project,
            name=name,
        )
        chief_client = mlrun.api.utils.clients.chief.Client()
        return await chief_client.delete_schedule(
            project=project, name=name, request=request
        )

    await run_in_threadpool(get_scheduler().delete_schedule, db_session, project, name)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.delete("/projects/{project}/schedules", status_code=HTTPStatus.NO_CONTENT.value)
async def delete_schedules(
    project: str,
    request: fastapi.Request,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    schedules = await run_in_threadpool(
        get_scheduler().list_schedules,
        db_session,
        project,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resources_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        schedules.schedules,
        lambda schedule: (schedule.project, schedule.name),
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_info,
    )
    # to reduce redundant load on the chief, we re-route the request only if the user has permissions
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to delete all project schedules, re-routing to chief",
            project=project,
        )
        chief_client = mlrun.api.utils.clients.chief.Client()
        return await chief_client.delete_schedules(project=project, request=request)

    await run_in_threadpool(get_scheduler().delete_schedules, db_session, project)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)
