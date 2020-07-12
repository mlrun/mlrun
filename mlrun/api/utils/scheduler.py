import asyncio
from typing import Any, Callable, List, Tuple, Dict, Union

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.utils.singletons.db import get_db
from mlrun.utils import logger


class Scheduler:
    # this should be something that does not make any sense to be inside project name or job name
    JOB_ID_SEPARATOR = "-_-"

    def __init__(self):
        self.scheduler = AsyncIOScheduler()

    async def start(self, db_session: Session):
        self.scheduler.start()
        # the scheduler shutdown and start operation are not fully async compatible yet -
        # https://github.com/agronholm/apscheduler/issues/360 - this sleep make them work
        await asyncio.sleep(0)
        self._reschedule_jobs(db_session)

    async def stop(self):
        self.scheduler.shutdown()
        # the scheduler shutdown and start operation are not fully async compatible yet -
        # https://github.com/agronholm/apscheduler/issues/360 - this sleep make them work
        await asyncio.sleep(0)

    def _verify_scheduler_down(self):
        if self.scheduler.running:
            raise RuntimeError('Scheduler still running')

    def create_schedule(
        self,
        db_session: Session,
        project: str,
        name: str,
        kind: schemas.ScheduledObjectKinds,
        scheduled_object: Union[Dict, Callable],
        cron_trigger: schemas.ScheduleCronTrigger,
    ):
        self._create_schedule_in_scheduler(
            db_session, project, name, kind, scheduled_object, cron_trigger
        )
        get_db().create_schedule(
            db_session, project, name, kind, scheduled_object, cron_trigger
        )

    def get_schedules(
        self, db_session: Session, project: str = None, kind: str = None
    ) -> schemas.Schedules:
        db_schedules = get_db().get_schedules(db_session, project, kind)
        schedules = []
        for db_schedule in db_schedules:
            schedule = self._transform_db_schedule_to_schedule(db_schedule)
            schedules.append(schedule)
        return schemas.Schedules(schedules=schedules)

    def get_schedule(
        self, db_session: Session, project: str, name: str
    ) -> schemas.Schedule:
        db_schedule = get_db().get_schedule(db_session, project, name)
        return self._transform_db_schedule_to_schedule(db_schedule)

    def delete_schedule(self, db_session: Session, project: str, name: str):
        job_id = self._resolve_job_identifier(project, name)
        self.scheduler.remove_job(job_id)
        get_db().delete_schedule(db_session, project, name)

    def _create_schedule_in_scheduler(
        self,
        db_session: Session,
        project: str,
        name: str,
        kind: schemas.ScheduledObjectKinds,
        scheduled_object: Any,
        cron_trigger: schemas.ScheduleCronTrigger,
    ):
        job_id = self._resolve_job_identifier(project, name)
        function, args, kwargs = self._resolve_job_function(
            db_session, kind, scheduled_object
        )
        self.scheduler.add_job(
            function, cron_trigger.to_apscheduler_cron_trigger(), args, kwargs, job_id
        )

    def _reschedule_jobs(self, db_session: Session):
        db_schedules = get_db().get_schedules(db_session)
        for db_schedule in db_schedules:
            self._create_schedule_in_scheduler(
                db_session,
                db_schedule.project,
                db_schedule.name,
                db_schedule.kind,
                db_schedule.scheduled_object,
                db_schedule.cron_trigger,
            )

    def _transform_db_schedule_to_schedule(
        self, db_schedule: schemas.ScheduleInDB
    ) -> schemas.Schedule:
        job_id = self._resolve_job_identifier(db_schedule.project, db_schedule.name)
        job = self.scheduler.get_job(job_id)
        schedule = schemas.Schedule(**db_schedule.dict())
        schedule.next_run_time = job.next_run_time
        return schedule

    @staticmethod
    def _resolve_job_identifier(project, name) -> str:
        """
        :return: returns the identifier that will be used inside the APScheduler
        """
        return Scheduler.JOB_ID_SEPARATOR.join([project, name])

    def _resolve_job_function(
        self,
        db_session: Session,
        scheduled_object_kind: schemas.ScheduledObjectKinds,
        scheduled_object: Any,
    ) -> Tuple[Callable, Union[List, Tuple], Dict]:
        """
        :return: a tuple (function, args, kwargs) to be used with the APScheduler.add_job
        """

        if scheduled_object_kind == schemas.ScheduledObjectKinds.job:
            from mlrun.api.api.utils import submit

            return submit, [db_session, scheduled_object], {}
        if scheduled_object_kind == schemas.ScheduledObjectKinds.pipeline:
            raise NotImplementedError("Pipeline scheduling Not implemented yet")
        if scheduled_object_kind == schemas.ScheduledObjectKinds.local_function:
            return scheduled_object, [], {}

        # sanity
        message = "Scheduled object kind missing implementation"
        logger.warn(message, scheduled_object_kind=scheduled_object_kind)
        raise NotImplementedError(message)
