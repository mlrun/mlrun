# Copyright 2023 Iguazio
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
import mlrun.api.crud
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.clients.chief
import mlrun.api.utils.singletons.project_member
import mlrun.common.schemas
from mlrun.api.api import deps
from mlrun.api.utils.singletons.scheduler import get_scheduler
from mlrun.utils import logger

router = APIRouter(prefix="/projects/{project}/schedules")


@router.post("")
async def create_schedule(
    project: str,
    schedule: mlrun.common.schemas.ScheduleInput,
    request: fastapi.Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.schedule,
        project,
        schedule.name,
        mlrun.common.schemas.AuthorizationAction.create,
        auth_info,
    )
    # to reduce redundant load on the chief, we re-route the request only if the user has permissions
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
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


@router.put("/{name}")
async def update_schedule(
    project: str,
    name: str,
    schedule: mlrun.common.schemas.ScheduleUpdate,
    request: fastapi.Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.update,
        auth_info,
    )
    # to reduce redundant load on the chief, we re-route the request only if the user has permissions
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
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


@router.get("", response_model=mlrun.common.schemas.SchedulesOutput)
async def list_schedules(
    project: str,
    name: str = None,
    labels: str = None,
    kind: mlrun.common.schemas.ScheduleKinds = None,
    include_last_run: bool = False,
    include_credentials: bool = fastapi.Query(False, alias="include-credentials"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
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
        mlrun.common.schemas.AuthorizationResourceTypes.schedule,
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
    "/{name}",
    response_model=mlrun.common.schemas.ScheduleOutput,
)
async def get_schedule(
    project: str,
    name: str,
    include_last_run: bool = False,
    include_credentials: bool = fastapi.Query(False, alias="include-credentials"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
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
        mlrun.common.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return schedule


@router.post("/{name}/invoke")
async def invoke_schedule(
    project: str,
    name: str,
    request: fastapi.Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.update,
        auth_info,
    )
    # to reduce redundant load on the chief, we re-route the request only if the user has permissions
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
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


@router.delete("/{name}", status_code=HTTPStatus.NO_CONTENT.value)
async def delete_schedule(
    project: str,
    name: str,
    request: fastapi.Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    # to reduce redundant load on the chief, we re-route the request only if the user has permissions
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
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


@router.delete("", status_code=HTTPStatus.NO_CONTENT.value)
async def delete_schedules(
    project: str,
    request: fastapi.Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    schedules = await run_in_threadpool(
        get_scheduler().list_schedules,
        db_session,
        project,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resources_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.schedule,
        schedules.schedules,
        lambda schedule: (schedule.project, schedule.name),
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    # to reduce redundant load on the chief, we re-route the request only if the user has permissions
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to delete all project schedules, re-routing to chief",
            project=project,
        )
        chief_client = mlrun.api.utils.clients.chief.Client()
        return await chief_client.delete_schedules(project=project, request=request)

    await run_in_threadpool(get_scheduler().delete_schedules, db_session, project)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.put("/{name}/notifications", status_code=HTTPStatus.OK.value)
async def set_schedule_notifications(
    project: str,
    name: str,
    request: fastapi.Request,
    set_notifications_request: mlrun.common.schemas.SetNotificationRequest = fastapi.Body(
        ...
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    # check permission per object type
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.schedule,
        project,
        resource_name=name,
        action=mlrun.common.schemas.AuthorizationAction.update,
        auth_info=auth_info,
    )

    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to set schedule notifications, re-routing to chief",
            project=project,
            schedule=set_notifications_request.dict(),
        )
        chief_client = mlrun.api.utils.clients.chief.Client()
        return await chief_client.set_schedule_notifications(
            project=project,
            schedule_name=name,
            request=request,
            json=set_notifications_request.dict(),
        )

    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.crud.Notifications().set_object_notifications,
        db_session,
        auth_info,
        project,
        set_notifications_request.notifications,
        mlrun.common.schemas.ScheduleIdentifier(name=name),
    )
    return fastapi.Response(status_code=HTTPStatus.OK.value)
