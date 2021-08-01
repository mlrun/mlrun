from http import HTTPStatus

import fastapi.concurrency
from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

import mlrun.api.utils.clients.opa
import mlrun.api.utils.singletons.project_member
from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.utils.singletons.scheduler import get_scheduler

router = APIRouter()


@router.post("/projects/{project}/schedules")
def create_schedule(
    project: str,
    schedule: schemas.ScheduleInput,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.singletons.project_member.get_project_member().ensure_project(
        db_session, project, auth_info=auth_verifier.auth_info
    )
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        project,
        schedule.name,
        mlrun.api.schemas.AuthorizationAction.create,
        auth_verifier.auth_info,
    )
    get_scheduler().create_schedule(
        db_session,
        auth_verifier.auth_info,
        project,
        schedule.name,
        schedule.kind,
        schedule.scheduled_object,
        schedule.cron_trigger,
        labels=schedule.labels,
        concurrency_limit=schedule.concurrency_limit,
    )
    return Response(status_code=HTTPStatus.CREATED.value)


@router.put("/projects/{project}/schedules/{name}")
def update_schedule(
    project: str,
    name: str,
    schedule: schemas.ScheduleUpdate,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.update,
        auth_verifier.auth_info,
    )
    get_scheduler().update_schedule(
        db_session,
        auth_verifier.auth_info,
        project,
        name,
        schedule.scheduled_object,
        schedule.cron_trigger,
        labels=schedule.labels,
    )
    return Response(status_code=HTTPStatus.OK.value)


@router.get("/projects/{project}/schedules", response_model=schemas.SchedulesOutput)
def list_schedules(
    project: str,
    name: str = None,
    labels: str = None,
    kind: schemas.ScheduleKinds = None,
    include_last_run: bool = False,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    schedules = get_scheduler().list_schedules(
        db_session, project, name, kind, labels, include_last_run=include_last_run,
    )
    filtered_schedules = mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        schedules.schedules,
        lambda schedule: (schedule.project, schedule.name,),
        auth_verifier.auth_info,
    )
    schedules.schedules = filtered_schedules
    return schedules


@router.get(
    "/projects/{project}/schedules/{name}", response_model=schemas.ScheduleOutput
)
def get_schedule(
    project: str,
    name: str,
    include_last_run: bool = False,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_verifier.auth_info,
    )
    return get_scheduler().get_schedule(
        db_session, project, name, include_last_run=include_last_run,
    )


@router.post("/projects/{project}/schedules/{name}/invoke")
async def invoke_schedule(
    project: str,
    name: str,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.utils.clients.opa.Client().query_resource_permissions,
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.update,
        auth_verifier.auth_info,
    )
    return await get_scheduler().invoke_schedule(
        db_session, auth_verifier.auth_info, project, name
    )


@router.delete(
    "/projects/{project}/schedules/{name}", status_code=HTTPStatus.NO_CONTENT.value
)
def delete_schedule(
    project: str,
    name: str,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_verifier.auth_info,
    )
    get_scheduler().delete_schedule(db_session, project, name)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.delete("/projects/{project}/schedules", status_code=HTTPStatus.NO_CONTENT.value)
def delete_schedules(
    project: str,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    schedules = get_scheduler().list_schedules(db_session, project,)
    mlrun.api.utils.clients.opa.Client().query_resources_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.schedule,
        schedules.schedules,
        lambda schedule: (schedule.project, schedule.name),
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_verifier.auth_info,
    )
    get_scheduler().delete_schedules(db_session, project)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)
