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
    )
    return Response(status_code=HTTPStatus.CREATED.value)


@router.get("/projects/{project}/schedules", response_model=schemas.SchedulesOutput)
def list_schedules(
    project: str,
    name: str = None,
    kind: schemas.ScheduleKinds = None,
    db_session: Session = Depends(deps.get_db_session),
):
    return get_scheduler().list_schedules(db_session, project, name, kind)


@router.get(
    "/projects/{project}/schedules/{name}", response_model=schemas.ScheduleOutput
)
def get_schedule(
    project: str, name: str, db_session: Session = Depends(deps.get_db_session),
):
    return get_scheduler().get_schedule(db_session, project, name)


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
