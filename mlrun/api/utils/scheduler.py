import asyncio
import copy
import json
import traceback
import typing
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import fastapi.concurrency
import humanfriendly
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger as APSchedulerCronTrigger
from sqlalchemy.orm import Session

import mlrun.api.utils.auth.verifier
import mlrun.errors
from mlrun.api import schemas
from mlrun.api.db.session import close_session, create_session
from mlrun.api.utils.singletons.db import get_db
from mlrun.config import config
from mlrun.model import RunObject
from mlrun.runtimes.constants import RunStates
from mlrun.utils import logger


class Scheduler:
    def __init__(self):
        scheduler_config = json.loads(config.httpdb.scheduling.scheduler_config)
        self._scheduler = AsyncIOScheduler(gconfig=scheduler_config, prefix=None)
        # this should be something that does not make any sense to be inside project name or job name
        self._job_id_separator = "-_-"
        # we don't allow to schedule a job to run more then one time per X
        # NOTE this cannot be less then one minute - see _validate_cron_trigger
        self._min_allowed_interval = config.httpdb.scheduling.min_allowed_interval
        self._secrets_provider = schemas.SecretProviderName.kubernetes

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
        auth_info: mlrun.api.schemas.AuthInfo,
        project: str,
        name: str,
        kind: schemas.ScheduleKinds,
        scheduled_object: Union[Dict, Callable],
        cron_trigger: Union[str, schemas.ScheduleCronTrigger],
        labels: Dict = None,
        concurrency_limit: int = None,
    ):
        if concurrency_limit is None:
            concurrency_limit = config.httpdb.scheduling.default_concurrency_limit
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
            concurrency_limit=concurrency_limit,
        )
        self._ensure_auth_info_has_access_key(auth_info, kind)
        self._store_schedule_secrets(auth_info, project, name)
        get_db().create_schedule(
            db_session,
            project,
            name,
            kind,
            scheduled_object,
            cron_trigger,
            concurrency_limit,
            labels,
        )
        self._create_schedule_in_scheduler(
            project,
            name,
            kind,
            scheduled_object,
            cron_trigger,
            concurrency_limit,
            auth_info,
        )

    def update_schedule(
        self,
        db_session: Session,
        auth_info: mlrun.api.schemas.AuthInfo,
        project: str,
        name: str,
        scheduled_object: Union[Dict, Callable] = None,
        cron_trigger: Union[str, schemas.ScheduleCronTrigger] = None,
        labels: Dict = None,
        concurrency_limit: int = None,
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
            concurrency_limit=concurrency_limit,
        )
        get_db().update_schedule(
            db_session,
            project,
            name,
            scheduled_object,
            cron_trigger,
            labels,
            concurrency_limit,
        )
        db_schedule = get_db().get_schedule(db_session, project, name)
        updated_schedule = self._transform_and_enrich_db_schedule(
            db_session, db_schedule
        )

        self._ensure_auth_info_has_access_key(auth_info, db_schedule.kind)
        self._store_schedule_secrets(auth_info, project, name)
        self._update_schedule_in_scheduler(
            project,
            name,
            updated_schedule.kind,
            updated_schedule.scheduled_object,
            updated_schedule.cron_trigger,
            updated_schedule.concurrency_limit,
            auth_info,
        )

    def list_schedules(
        self,
        db_session: Session,
        project: str = None,
        name: str = None,
        kind: str = None,
        labels: str = None,
        include_last_run: bool = False,
        include_credentials: bool = False,
    ) -> schemas.SchedulesOutput:
        logger.debug(
            "Getting schedules", project=project, name=name, labels=labels, kind=kind
        )
        db_schedules = get_db().list_schedules(db_session, project, name, labels, kind)
        schedules = []
        for db_schedule in db_schedules:
            schedule = self._transform_and_enrich_db_schedule(
                db_session, db_schedule, include_last_run, include_credentials
            )
            schedules.append(schedule)
        return schemas.SchedulesOutput(schedules=schedules)

    def get_schedule(
        self,
        db_session: Session,
        project: str,
        name: str,
        include_last_run: bool = False,
        include_credentials: bool = False,
    ) -> schemas.ScheduleOutput:
        logger.debug("Getting schedule", project=project, name=name)
        db_schedule = get_db().get_schedule(db_session, project, name)
        return self._transform_and_enrich_db_schedule(
            db_session, db_schedule, include_last_run, include_credentials
        )

    def delete_schedule(
        self, db_session: Session, project: str, name: str,
    ):
        logger.debug("Deleting schedule", project=project, name=name)
        self._remove_schedule_scheduler_resources(project, name)
        get_db().delete_schedule(db_session, project, name)

    def delete_schedules(
        self, db_session: Session, project: str,
    ):
        schedules = self.list_schedules(db_session, project,)
        logger.debug("Deleting schedules", project=project)
        for schedule in schedules.schedules:
            self._remove_schedule_scheduler_resources(schedule.project, schedule.name)
        get_db().delete_schedules(db_session, project)

    def _remove_schedule_scheduler_resources(self, project, name):
        self._remove_schedule_from_scheduler(project, name)
        self._remove_schedule_secrets(project, name)

    def _remove_schedule_from_scheduler(self, project, name):
        job_id = self._resolve_job_id(project, name)
        # don't fail on delete if job doesn't exist
        job = self._scheduler.get_job(job_id)
        if job:
            self._scheduler.remove_job(job_id)

    async def invoke_schedule(
        self,
        db_session: Session,
        auth_info: mlrun.api.schemas.AuthInfo,
        project: str,
        name: str,
    ):
        logger.debug("Invoking schedule", project=project, name=name)
        db_schedule = await fastapi.concurrency.run_in_threadpool(
            get_db().get_schedule, db_session, project, name
        )
        function, args, kwargs = self._resolve_job_function(
            db_schedule.kind,
            db_schedule.scheduled_object,
            project,
            name,
            db_schedule.concurrency_limit,
            auth_info,
        )
        return await function(*args, **kwargs)

    def _ensure_auth_info_has_access_key(
        self, auth_info: mlrun.api.schemas.AuthInfo, kind: schemas.ScheduleKinds,
    ):
        if (
            kind not in schemas.ScheduleKinds.local_kinds()
            and mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required()
            and (
                not auth_info.access_key
                or auth_info.access_key == mlrun.model.Credentials.generate_access_key
            )
        ):
            auth_info.access_key = mlrun.api.utils.auth.verifier.AuthVerifier().get_or_create_access_key(
                auth_info.session
            )

    def _store_schedule_secrets(
        self, auth_info: mlrun.api.schemas.AuthInfo, project: str, name: str,
    ):
        # import here to avoid circular imports
        import mlrun.api.crud

        if mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required():
            # sanity
            if not auth_info.access_key:
                raise mlrun.errors.MLRunAccessDeniedError(
                    "Access key is required to create schedules in OPA authorization mode"
                )
            access_key_secret_key = mlrun.api.crud.Secrets().generate_schedule_access_key_secret_key(
                name
            )
            # schedule name may be an invalid secret key, therefore we're using the key map feature of our secrets
            # handler
            secret_key_map = (
                mlrun.api.crud.Secrets().generate_schedule_key_map_secret_key()
            )
            secrets = {
                access_key_secret_key: auth_info.access_key,
            }
            if auth_info.username:
                username_secret_key = mlrun.api.crud.Secrets().generate_schedule_username_secret_key(
                    name
                )
                secrets[username_secret_key] = auth_info.username
            mlrun.api.crud.Secrets().store_secrets(
                project,
                schemas.SecretsData(provider=self._secrets_provider, secrets=secrets,),
                allow_internal_secrets=True,
                key_map_secret_key=secret_key_map,
            )

    def _remove_schedule_secrets(
        self, project: str, name: str,
    ):
        # import here to avoid circular imports
        import mlrun.api.crud

        if mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required():
            access_key_secret_key = mlrun.api.crud.Secrets().generate_schedule_access_key_secret_key(
                name
            )
            username_secret_key = mlrun.api.crud.Secrets().generate_schedule_username_secret_key(
                name
            )
            secret_key_map = (
                mlrun.api.crud.Secrets().generate_schedule_key_map_secret_key()
            )
            # TODO: support delete secrets (plural and not only singular) using key map
            mlrun.api.crud.Secrets().delete_secret(
                project,
                self._secrets_provider,
                access_key_secret_key,
                allow_secrets_from_k8s=True,
                allow_internal_secrets=True,
                key_map_secret_key=secret_key_map,
            )
            mlrun.api.crud.Secrets().delete_secret(
                project,
                self._secrets_provider,
                username_secret_key,
                allow_secrets_from_k8s=True,
                allow_internal_secrets=True,
                key_map_secret_key=secret_key_map,
            )

    def _get_schedule_secrets(
        self, project: str, name: str, include_username: bool = True
    ) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]:
        # import here to avoid circular imports
        import mlrun.api.crud

        schedule_access_key_secret_key = mlrun.api.crud.Secrets().generate_schedule_access_key_secret_key(
            name
        )
        secret_key_map = mlrun.api.crud.Secrets().generate_schedule_key_map_secret_key()
        # TODO: support listing (and not only get) secrets using key map
        access_key = mlrun.api.crud.Secrets().get_secret(
            project,
            self._secrets_provider,
            schedule_access_key_secret_key,
            allow_secrets_from_k8s=True,
            allow_internal_secrets=True,
            key_map_secret_key=secret_key_map,
        )
        username = None
        if include_username:
            schedule_username_secret_key = mlrun.api.crud.Secrets().generate_schedule_username_secret_key(
                name
            )
            username = mlrun.api.crud.Secrets().get_secret(
                project,
                self._secrets_provider,
                schedule_username_secret_key,
                allow_secrets_from_k8s=True,
                allow_internal_secrets=True,
                key_map_secret_key=secret_key_map,
            )

        return username, access_key

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
        concurrency_limit: int,
        auth_info: mlrun.api.schemas.AuthInfo,
    ):
        job_id = self._resolve_job_id(project, name)
        logger.debug("Adding schedule to scheduler", job_id=job_id)
        function, args, kwargs = self._resolve_job_function(
            kind, scheduled_object, project, name, concurrency_limit, auth_info
        )

        # we use max_instances as well as our logic in the run wrapper for concurrent jobs
        # in order to allow concurrency for triggering the jobs (max_instances), and concurrency
        # of the jobs themselves (our logic in the run wrapper).
        self._scheduler.add_job(
            function,
            self.transform_schemas_cron_trigger_to_apscheduler_cron_trigger(
                cron_trigger
            ),
            args,
            kwargs,
            job_id,
            max_instances=concurrency_limit,
        )

    def _update_schedule_in_scheduler(
        self,
        project: str,
        name: str,
        kind: schemas.ScheduleKinds,
        scheduled_object: Any,
        cron_trigger: schemas.ScheduleCronTrigger,
        concurrency_limit: int,
        auth_info: mlrun.api.schemas.AuthInfo,
    ):
        job_id = self._resolve_job_id(project, name)
        logger.debug("Updating schedule in scheduler", job_id=job_id)
        function, args, kwargs = self._resolve_job_function(
            kind, scheduled_object, project, name, concurrency_limit, auth_info
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
                # import here to avoid circular imports
                import mlrun.api.crud

                access_key = None
                username = None
                if mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required():
                    username, access_key = self._get_schedule_secrets(
                        db_schedule.project, db_schedule.name
                    )
                self._create_schedule_in_scheduler(
                    db_schedule.project,
                    db_schedule.name,
                    db_schedule.kind,
                    db_schedule.scheduled_object,
                    db_schedule.cron_trigger,
                    db_schedule.concurrency_limit,
                    mlrun.api.schemas.AuthInfo(
                        username=username, access_key=access_key
                    ),
                )
            except Exception as exc:
                logger.warn(
                    "Failed rescheduling job. Continuing",
                    exc=str(exc),
                    traceback=traceback.format_exc(),
                    db_schedule=db_schedule,
                )

    def _transform_and_enrich_db_schedule(
        self,
        db_session: Session,
        schedule_record: schemas.ScheduleRecord,
        include_last_run: bool = False,
        include_credentials: bool = False,
    ) -> schemas.ScheduleOutput:
        schedule_dict = schedule_record.dict()
        schedule_dict["labels"] = {
            label["name"]: label["value"] for label in schedule_dict["labels"]
        }
        schedule = schemas.ScheduleOutput(**schedule_dict)

        job_id = self._resolve_job_id(schedule_record.project, schedule_record.name)
        job = self._scheduler.get_job(job_id)
        if job:
            schedule.next_run_time = job.next_run_time

        if include_last_run:
            self._enrich_schedule_with_last_run(db_session, schedule)

        if include_credentials:
            self._enrich_schedule_with_credentials(schedule)

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

    def _enrich_schedule_with_credentials(
        self, schedule_output: schemas.ScheduleOutput
    ):
        _, schedule_output.credentials.access_key = self._get_schedule_secrets(
            schedule_output.project, schedule_output.name, include_username=False
        )

    def _resolve_job_function(
        self,
        scheduled_kind: schemas.ScheduleKinds,
        scheduled_object: Any,
        project_name: str,
        schedule_name: str,
        schedule_concurrency_limit: int,
        auth_info: mlrun.api.schemas.AuthInfo,
    ) -> Tuple[Callable, Optional[Union[List, Tuple]], Optional[Dict]]:
        """
        :return: a tuple (function, args, kwargs) to be used with the APScheduler.add_job
        """

        if scheduled_kind == schemas.ScheduleKinds.job:
            scheduled_object_copy = copy.deepcopy(scheduled_object)
            return (
                Scheduler.submit_run_wrapper,
                [
                    self,
                    scheduled_object_copy,
                    project_name,
                    schedule_name,
                    schedule_concurrency_limit,
                    auth_info,
                ],
                {},
            )
        if scheduled_kind == schemas.ScheduleKinds.local_function:
            return scheduled_object, [], {}

        # sanity
        message = "Scheduled object kind missing implementation"
        logger.warn(message, scheduled_object_kind=scheduled_kind)
        raise NotImplementedError(message)

    def _list_schedules_from_scheduler(self, project: str):
        jobs = self._scheduler.get_jobs()
        return [job for job in jobs if self._resolve_job_id(project, "") in job.id]

    def _resolve_job_id(self, project, name) -> str:
        """
        :return: returns the identifier that will be used inside the APScheduler
        """
        return self._job_id_separator.join([project, name])

    @staticmethod
    async def submit_run_wrapper(
        scheduler,
        scheduled_object,
        project_name,
        schedule_name,
        schedule_concurrency_limit,
        auth_info: mlrun.api.schemas.AuthInfo,
    ):
        # import here to avoid circular imports
        import mlrun.api.crud
        from mlrun.api.api.utils import submit_run

        # removing the schedule from the body otherwise when the scheduler will submit this task it will go to an
        # endless scheduling loop
        scheduled_object.pop("schedule", None)

        # removing the uid from the task metadata so that a new uid will be generated for every run
        # otherwise all runs will have the same uid
        scheduled_object.get("task", {}).get("metadata", {}).pop("uid", None)

        if "task" in scheduled_object and "metadata" in scheduled_object["task"]:
            scheduled_object["task"]["metadata"].setdefault("labels", {})
            scheduled_object["task"]["metadata"]["labels"][
                schemas.constants.LabelNames.schedule_name
            ] = schedule_name

        db_session = create_session()

        active_runs = mlrun.api.crud.Runs().list_runs(
            db_session,
            states=RunStates.non_terminal_states(),
            project=project_name,
            labels=f"{schemas.constants.LabelNames.schedule_name}={schedule_name}",
        )
        if len(active_runs) >= schedule_concurrency_limit:
            logger.warn(
                "Schedule exceeded concurrency limit, skipping this run",
                project=project_name,
                schedule_name=schedule_name,
                schedule_concurrency_limit=schedule_concurrency_limit,
                active_runs=len(active_runs),
            )
            return

        # if credentials are needed but missing (will happen for schedules on upgrade from scheduler that didn't store
        # credentials to one that does store) enrich them
        # Note that here we're using the "knowledge" that submit_run only requires the access key of the auth info
        if (
            not auth_info.access_key
            and mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required()
        ):
            # import here to avoid circular imports
            import mlrun.api.utils.auth
            import mlrun.api.utils.singletons.project_member

            logger.info(
                "Schedule missing auth info which is required. Trying to fill from project owner",
                project_name=project_name,
                schedule_name=schedule_name,
            )

            project_owner = mlrun.api.utils.singletons.project_member.get_project_member().get_project_owner(
                db_session, project_name
            )
            # Update the schedule with the new auth info so we won't need to do the above again in the next run
            scheduler.update_schedule(
                db_session,
                mlrun.api.schemas.AuthInfo(
                    username=project_owner.username, access_key=project_owner.session
                ),
                project_name,
                schedule_name,
            )

        response = await submit_run(db_session, auth_info, scheduled_object)

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
