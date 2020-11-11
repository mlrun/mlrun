import asyncio
import copy
from datetime import datetime, timedelta
from typing import Any, Callable, List, Tuple, Dict, Union, Optional

import humanfriendly
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger as APSchedulerCronTrigger
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.db.session import create_session, close_session
from mlrun.api.utils.singletons.db import get_db
from mlrun.model import RunObject
from mlrun.config import config
from mlrun.utils import logger


class Scheduler:
    def __init__(self):
        self._scheduler = AsyncIOScheduler()
        # this should be something that does not make any sense to be inside project name or job name
        self._job_id_separator = "-_-"
        # we don't allow to schedule a job to run more then one time per X
        # NOTE this cannot be less then one minute - see _validate_cron_trigger
        self._min_allowed_interval = config.httpdb.scheduling.min_allowed_interval

    async def start(self, db_session: Session):
        logger.info("Starting scheduler")
        self._scheduler.start()
        # the scheduler shutdown and start operation are not fully async compatible yet -
        # https://github.com/agronholm/apscheduler/issues/360 - this sleep make them work
        await asyncio.sleep(0)

        # don't fail the start on re-scheduling failure
        try:
            self._reload_schedules(db_session)
        except Exception as exc:
            logger.warning("Failed reloading schedules", exc=exc)

    async def stop(self):
        logger.info("Stopping scheduler")
        self._scheduler.shutdown()
        # the scheduler shutdown and start operation are not fully async compatible yet -
        # https://github.com/agronholm/apscheduler/issues/360 - this sleep make them work
        await asyncio.sleep(0)

    def create_schedule(
        self,
        db_session: Session,
        project: str,
        name: str,
        kind: schemas.ScheduleKinds,
        scheduled_object: Union[Dict, Callable],
        cron_trigger: Union[str, schemas.ScheduleCronTrigger],
        labels: Dict = None,
    ):
        if isinstance(cron_trigger, str):
            cron_trigger = schemas.ScheduleCronTrigger.from_crontab(cron_trigger)

        self._validate_cron_trigger(cron_trigger)

        logger.debug(
            "Creating schedule",
            project=project,
            name=name,
            kind=kind,
            scheduled_object=scheduled_object,
            cron_trigger=cron_trigger,
            labels=labels,
        )
        get_db().create_schedule(
            db_session, project, name, kind, scheduled_object, cron_trigger, labels
        )
        self._create_schedule_in_scheduler(
            project, name, kind, scheduled_object, cron_trigger
        )

    def update_schedule(
        self,
        db_session: Session,
        project: str,
        name: str,
        scheduled_object: Union[Dict, Callable] = None,
        cron_trigger: Union[str, schemas.ScheduleCronTrigger] = None,
        labels: Dict = None,
    ):
        if isinstance(cron_trigger, str):
            cron_trigger = schemas.ScheduleCronTrigger.from_crontab(cron_trigger)

        if cron_trigger is not None:
            self._validate_cron_trigger(cron_trigger)

        logger.debug(
            "Updating schedule",
            project=project,
            name=name,
            scheduled_object=scheduled_object,
            cron_trigger=cron_trigger,
            labels=labels,
        )
        get_db().update_schedule(
            db_session, project, name, scheduled_object, cron_trigger, labels
        )
        db_schedule = get_db().get_schedule(db_session, project, name)
        updated_schedule = self._transform_and_enrich_db_schedule(
            db_session, db_schedule
        )

        self._update_schedule_in_scheduler(
            project,
            name,
            updated_schedule.kind,
            updated_schedule.scheduled_object,
            updated_schedule.cron_trigger,
        )

    def list_schedules(
        self,
        db_session: Session,
        project: str = None,
        name: str = None,
        kind: str = None,
        labels: str = None,
        include_last_run: bool = False,
    ) -> schemas.SchedulesOutput:
        logger.debug(
            "Getting schedules", project=project, name=name, labels=labels, kind=kind
        )
        db_schedules = get_db().list_schedules(db_session, project, name, labels, kind)
        schedules = []
        for db_schedule in db_schedules:
            schedule = self._transform_and_enrich_db_schedule(
                db_session, db_schedule, include_last_run
            )
            schedules.append(schedule)
        return schemas.SchedulesOutput(schedules=schedules)

    def get_schedule(
        self,
        db_session: Session,
        project: str,
        name: str,
        include_last_run: bool = False,
    ) -> schemas.ScheduleOutput:
        logger.debug("Getting schedule", project=project, name=name)
        db_schedule = get_db().get_schedule(db_session, project, name)
        return self._transform_and_enrich_db_schedule(
            db_session, db_schedule, include_last_run
        )

    def delete_schedule(self, db_session: Session, project: str, name: str):
        logger.debug("Deleting schedule", project=project, name=name)
        job_id = self._resolve_job_id(project, name)
        self._scheduler.remove_job(job_id)
        get_db().delete_schedule(db_session, project, name)

    async def invoke_schedule(self, db_session: Session, project: str, name: str):
        logger.debug("Invoking schedule", project=project, name=name)
        db_schedule = get_db().get_schedule(db_session, project, name)
        function, args, kwargs = self._resolve_job_function(
            db_schedule.kind, db_schedule.scheduled_object, name
        )
        return await function(*args, **kwargs)

    def _validate_cron_trigger(
        self,
        cron_trigger: schemas.ScheduleCronTrigger,
        # accepting now from outside for testing purposes
        now: datetime = None,
    ):
        """
        Enforce no more then one job per min_allowed_interval
        """
        logger.debug("Validating cron trigger")
        apscheduler_cron_trigger = self.transform_schemas_cron_trigger_to_apscheduler_cron_trigger(
            cron_trigger
        )
        now = now or datetime.now(apscheduler_cron_trigger.timezone)
        next_run_time = None
        second_next_run_time = now

        # doing 60 checks to allow one minute precision, if the _min_allowed_interval is less then one minute validation
        # won't fail in certain scenarios that it should. See test_validate_cron_trigger_multi_checks for detailed
        # explanation
        for index in range(60):
            next_run_time = apscheduler_cron_trigger.get_next_fire_time(
                None, second_next_run_time
            )
            # will be none if we got a schedule that has no next fire time - for example schedule with year=1999
            if next_run_time is None:
                return
            second_next_run_time = apscheduler_cron_trigger.get_next_fire_time(
                next_run_time, next_run_time
            )
            # will be none if we got a schedule that has no next fire time - for example schedule with year=2050
            if second_next_run_time is None:
                return
            min_allowed_interval_seconds = humanfriendly.parse_timespan(
                self._min_allowed_interval
            )
            if second_next_run_time < next_run_time + timedelta(
                seconds=min_allowed_interval_seconds
            ):
                logger.warn(
                    "Cron trigger too frequent. Rejecting",
                    cron_trigger=cron_trigger,
                    next_run_time=next_run_time,
                    second_next_run_time=second_next_run_time,
                    delta=second_next_run_time - next_run_time,
                )
                raise ValueError(
                    f"Cron trigger too frequent. no more then one job "
                    f"per {self._min_allowed_interval} is allowed"
                )

    def _create_schedule_in_scheduler(
        self,
        project: str,
        name: str,
        kind: schemas.ScheduleKinds,
        scheduled_object: Any,
        cron_trigger: schemas.ScheduleCronTrigger,
    ):
        job_id = self._resolve_job_id(project, name)
        logger.debug("Adding schedule to scheduler", job_id=job_id)
        function, args, kwargs = self._resolve_job_function(
            kind, scheduled_object, name
        )
        self._scheduler.add_job(
            function,
            self.transform_schemas_cron_trigger_to_apscheduler_cron_trigger(
                cron_trigger
            ),
            args,
            kwargs,
            job_id,
        )

    def _update_schedule_in_scheduler(
        self,
        project: str,
        name: str,
        kind: schemas.ScheduleKinds,
        scheduled_object: Any,
        cron_trigger: schemas.ScheduleCronTrigger,
    ):
        job_id = self._resolve_job_id(project, name)
        logger.debug("Updating schedule in scheduler", job_id=job_id)
        function, args, kwargs = self._resolve_job_function(
            kind, scheduled_object, name
        )
        trigger = self.transform_schemas_cron_trigger_to_apscheduler_cron_trigger(
            cron_trigger
        )
        now = datetime.now(self._scheduler.timezone)
        next_run_time = trigger.get_next_fire_time(None, now)
        self._scheduler.modify_job(
            job_id,
            func=function,
            args=args,
            kwargs=kwargs,
            trigger=trigger,
            next_run_time=next_run_time,
        )

    def _reload_schedules(self, db_session: Session):
        logger.info("Reloading schedules")
        db_schedules = get_db().list_schedules(db_session)
        for db_schedule in db_schedules:
            # don't let one failure fail the rest
            try:
                self._create_schedule_in_scheduler(
                    db_schedule.project,
                    db_schedule.name,
                    db_schedule.kind,
                    db_schedule.scheduled_object,
                    db_schedule.cron_trigger,
                )
            except Exception as exc:
                logger.warn(
                    "Failed rescheduling job. Continuing",
                    exc=str(exc),
                    db_schedule=db_schedule,
                )

    def _transform_and_enrich_db_schedule(
        self,
        db_session: Session,
        schedule_record: schemas.ScheduleRecord,
        include_last_run: bool = False,
    ) -> schemas.ScheduleOutput:
        schedule = schemas.ScheduleOutput(**schedule_record.dict())

        job_id = self._resolve_job_id(schedule_record.project, schedule_record.name)
        job = self._scheduler.get_job(job_id)
        schedule.next_run_time = job.next_run_time

        if include_last_run:
            schedule = self._enrich_schedule_with_last_run(db_session, schedule)

        return schedule

    @staticmethod
    def _enrich_schedule_with_last_run(
        db_session: Session, schedule_output: schemas.ScheduleOutput
    ):
        if schedule_output.last_run_uri:
            run_project, run_uid, iteration, _ = RunObject.parse_uri(
                schedule_output.last_run_uri
            )
            run_data = get_db().read_run(db_session, run_uid, run_project, iteration)
            schedule_output.last_run = run_data
        return schedule_output

    def _resolve_job_function(
        self,
        scheduled_kind: schemas.ScheduleKinds,
        scheduled_object: Any,
        schedule_name: str,
    ) -> Tuple[Callable, Optional[Union[List, Tuple]], Optional[Dict]]:
        """
        :return: a tuple (function, args, kwargs) to be used with the APScheduler.add_job
        """

        if scheduled_kind == schemas.ScheduleKinds.job:
            scheduled_object_copy = copy.deepcopy(scheduled_object)
            return (
                Scheduler.submit_run_wrapper,
                [scheduled_object_copy, schedule_name],
                {},
            )
        if scheduled_kind == schemas.ScheduleKinds.local_function:
            return scheduled_object, [], {}

        # sanity
        message = "Scheduled object kind missing implementation"
        logger.warn(message, scheduled_object_kind=scheduled_kind)
        raise NotImplementedError(message)

    def _resolve_job_id(self, project, name) -> str:
        """
        :return: returns the identifier that will be used inside the APScheduler
        """
        return self._job_id_separator.join([project, name])

    @staticmethod
    async def submit_run_wrapper(scheduled_object, schedule_name):
        # import here to avoid circular imports
        from mlrun.api.api.utils import submit_run

        # removing the schedule from the body otherwise when the scheduler will submit this task it will go to an
        # endless scheduling loop
        scheduled_object.pop("schedule", None)

        # removing the uid from the task metadata so that a new uid will be generated for every run
        # otherwise all runs will have the same uid
        scheduled_object.get("task", {}).get("metadata", {}).pop("uid", None)

        db_session = create_session()

        response = await submit_run(db_session, scheduled_object)

        run_metadata = response["data"]["metadata"]
        run_uri = RunObject.create_uri(
            run_metadata["project"], run_metadata["uid"], run_metadata["iteration"]
        )
        get_db().update_schedule(
            db_session, run_metadata["project"], schedule_name, last_run_uri=run_uri,
        )

        close_session(db_session)

        return response

    @staticmethod
    def transform_schemas_cron_trigger_to_apscheduler_cron_trigger(
        cron_trigger: schemas.ScheduleCronTrigger,
    ):
        return APSchedulerCronTrigger(
            cron_trigger.year,
            cron_trigger.month,
            cron_trigger.day,
            cron_trigger.week,
            cron_trigger.day_of_week,
            cron_trigger.hour,
            cron_trigger.minute,
            cron_trigger.second,
            cron_trigger.start_date,
            cron_trigger.end_date,
            cron_trigger.timezone,
            cron_trigger.jitter,
        )
