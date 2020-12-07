from http import HTTPStatus

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.utils.singletons.scheduler import get_scheduler

router = APIRouter()


@router.post("/projects/{project}/schedules")
def create_schedule(
    project: str,
    schedule: schemas.ScheduleInput,
    db_session: Session = Depends(deps.get_db_session),
):
    get_scheduler().create_schedule(
        db_session,
        project,
        schedule.name,
        schedule.kind,
        schedule.scheduled_object,
        schedule.cron_trigger,
        labels=schedule.labels,
    )
    return Response(status_code=HTTPStatus.CREATED.value)


@router.put("/projects/{project}/schedules/{name}")
def update_schedule(
    project: str,
    name: str,
    schedule: schemas.ScheduleUpdate,
    db_session: Session = Depends(deps.get_db_session),
):
    get_scheduler().update_schedule(
        db_session,
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
    db_session: Session = Depends(deps.get_db_session),
):
    return get_scheduler().list_schedules(
        db_session, project, name, kind, labels, include_last_run=include_last_run
    )


@router.get(
    "/projects/{project}/schedules/{name}", response_model=schemas.ScheduleOutput
)
def get_schedule(
    project: str,
    name: str,
    include_last_run: bool = False,
    db_session: Session = Depends(deps.get_db_session),
):
    return get_scheduler().get_schedule(
        db_session, project, name, include_last_run=include_last_run
    )


@router.post("/projects/{project}/schedules/{name}/invoke")
async def invoke_schedule(
    project: str, name: str, db_session: Session = Depends(deps.get_db_session),
):
    return await get_scheduler().invoke_schedule(db_session, project, name)


@router.delete(
    "/projects/{project}/schedules/{name}", status_code=HTTPStatus.NO_CONTENT.value
)
def delete_schedule(
    project: str, name: str, db_session: Session = Depends(deps.get_db_session),
):
    get_scheduler().delete_schedule(db_session, project, name)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)
