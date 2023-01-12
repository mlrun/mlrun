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
import asyncio
import pathlib
import random
import time
import typing
import unittest.mock
from datetime import datetime, timedelta, timezone

import pytest
from dateutil.tz import tzlocal
from deepdiff import DeepDiff
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.utils.auth
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.k8s
import mlrun.api.utils.singletons.project_member
import mlrun.errors
import tests.api.conftest
from mlrun.api import schemas
from mlrun.api.utils.scheduler import Scheduler
from mlrun.api.utils.singletons.db import get_db
from mlrun.config import config
from mlrun.runtimes.base import RunStates
from mlrun.utils import logger


@pytest.fixture()
async def scheduler(db: Session) -> typing.Generator:
    logger.info("Creating scheduler")
    config.httpdb.scheduling.min_allowed_interval = "0"
    config.httpdb.jobs.allow_local_run = True
    scheduler = Scheduler()
    await scheduler.start(db)
    mlrun.api.utils.singletons.project_member.initialize_project_member()
    yield scheduler
    logger.info("Stopping scheduler")
    await scheduler.stop()


call_counter: int = 0
schedule_end_time_margin = 0.5


async def bump_counter():
    global call_counter
    call_counter += 1


async def bump_counter_and_wait():
    global call_counter
    logger.debug("Bumping counter", call_counter=call_counter)
    call_counter += 1
    await asyncio.sleep(2)


async def do_nothing():
    pass


@pytest.mark.asyncio
async def test_not_skipping_delayed_schedules(db: Session, scheduler: Scheduler):
    global call_counter
    call_counter = 0
    expected_call_counter = 1

    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=expected_call_counter, seconds_interval=1
    )
    # this way we're leaving ourselves one second to create the schedule preventing transient test failure
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1", start_date=start_date, end_date=end_date
    )
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        bump_counter,
        cron_trigger,
    )
    # purposely doing time.sleep to block the reactor to ensure a job is still scheduled although its planned
    # execution time passed
    time.sleep(2 + expected_call_counter)
    await asyncio.sleep(1)
    assert call_counter == expected_call_counter


@pytest.mark.asyncio
async def test_create_schedule(db: Session, scheduler: Scheduler):
    global call_counter
    call_counter = 0

    expected_call_counter = 5
    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=5, seconds_interval=1
    )
    # this way we're leaving ourselves one second to create the schedule preventing transient test failure
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1", start_date=start_date, end_date=end_date
    )
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        bump_counter,
        cron_trigger,
    )

    # The trigger is defined with `second="*/1"` meaning it runs on round seconds,
    # but executing the actual functional code - bumping the counter - happens a few microseconds afterwards.
    # To avoid transient errors on slow systems, we add extra margin.
    time_to_sleep = (
        end_date - datetime.now()
    ).total_seconds() + schedule_end_time_margin

    await asyncio.sleep(time_to_sleep)
    assert call_counter == expected_call_counter


@pytest.mark.asyncio
async def test_invoke_schedule(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    cron_trigger = schemas.ScheduleCronTrigger(year=1999)
    schedule_name = "schedule-name"
    project_name = config.default_project
    mlrun.new_project(project_name, save=False)
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project_name,
        schedule_name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0
    response_1 = await scheduler.invoke_schedule(
        db, mlrun.api.schemas.AuthInfo(), project_name, schedule_name
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 1
    response_2 = await scheduler.invoke_schedule(
        db, mlrun.api.schemas.AuthInfo(), project_name, schedule_name
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 2
    for run in runs:
        assert run["status"]["state"] == RunStates.completed
    response_uids = [
        response["data"]["metadata"]["uid"] for response in [response_1, response_2]
    ]
    db_uids = [run["metadata"]["uid"] for run in runs]
    assert (
        DeepDiff(
            response_uids,
            db_uids,
            ignore_order=True,
        )
        == {}
    )

    schedule = scheduler.get_schedule(
        db, project_name, schedule_name, include_last_run=True
    )
    assert schedule.last_run is not None
    assert schedule.last_run["metadata"]["uid"] == response_uids[-1]
    assert schedule.last_run["metadata"]["project"] == project_name


@pytest.mark.asyncio
async def test_create_schedule_mlrun_function(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):

    expected_call_counter = 1
    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=expected_call_counter, seconds_interval=1
    )
    # this way we're leaving ourselves one second to create the schedule preventing transient test failure
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1", start_date=start_date, end_date=end_date
    )
    schedule_name = "schedule-name"
    project_name = config.default_project
    mlrun.new_project(project_name, save=False)

    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project_name,
        schedule_name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )
    time_to_sleep = (
        end_date - datetime.now()
    ).total_seconds() + schedule_end_time_margin

    await asyncio.sleep(time_to_sleep)
    runs = get_db().list_runs(db, project=project_name)

    assert len(runs) == expected_call_counter

    assert runs[0]["status"]["state"] == RunStates.completed

    # the default of list_runs returns the list descending by date.
    expected_last_run_uri = f"{project_name}@{runs[0]['metadata']['uid']}#0"

    schedule = get_db().get_schedule(db, project_name, schedule_name)
    assert schedule.last_run_uri == expected_last_run_uri


@pytest.mark.asyncio
async def test_create_schedule_success_cron_trigger_validation(
    db: Session, scheduler: Scheduler
):
    scheduler._min_allowed_interval = "10 minutes"
    cases = [
        {"second": "1", "minute": "19"},
        {"second": "30", "minute": "9,19"},
        {"minute": "*/10"},
        {"minute": "20-40/10"},
        {"hour": "1"},
        {"year": "1999"},
        {"year": "2050"},
    ]
    for index, case in enumerate(cases):
        cron_trigger = schemas.ScheduleCronTrigger(**case)
        scheduler.create_schedule(
            db,
            mlrun.api.schemas.AuthInfo(),
            "project",
            f"schedule-name-{index}",
            schemas.ScheduleKinds.local_function,
            do_nothing,
            cron_trigger,
        )


@pytest.mark.asyncio
async def test_schedule_upgrade_from_scheduler_without_credentials_store(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    name = "schedule-name"
    project_name = config.default_project
    mlrun.new_project(project_name, save=False)

    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name
    )

    expected_call_counter = 3
    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=expected_call_counter, seconds_interval=1
    )
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1", start_date=start_date, end_date=end_date
    )
    # we're before upgrade so create a schedule with empty auth info
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project_name,
        name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )
    # stop scheduler, reconfigure to store credentials and start again (upgrade)
    await scheduler.stop()
    mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    await scheduler.start(db)

    # at this point the schedule is inside the scheduler without auth_info, so the first trigger should try to generate
    # auth info, mock the functions for this
    username = "some-username"
    access_key = "some-access_key"
    mlrun.api.utils.singletons.project_member.get_project_member().get_project_owner = (
        unittest.mock.Mock(
            return_value=mlrun.api.schemas.ProjectOwner(
                username=username, access_key=access_key
            )
        )
    )
    time_to_sleep = (
        end_date - datetime.now()
    ).total_seconds() + schedule_end_time_margin

    await asyncio.sleep(time_to_sleep)
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 3
    assert (
        mlrun.api.utils.singletons.project_member.get_project_member().get_project_owner.call_count
        == 1
    )


@pytest.mark.asyncio
async def test_create_schedule_failure_too_frequent_cron_trigger(
    db: Session, scheduler: Scheduler
):
    scheduler._min_allowed_interval = "10 minutes"
    cases = [
        {"second": "*"},
        {"second": "1,2"},
        {"second": "*/30"},
        {"second": "30-35"},
        {"second": "30-40/5"},
        {"minute": "*"},
        {"minute": "*"},
        {"minute": "*/5"},
        {"minute": "43-59"},
        {"minute": "30-50/6"},
        {"minute": "1,3,5"},
        {"minute": "11,22,33,44,55,59"},
    ]
    for case in cases:
        cron_trigger = schemas.ScheduleCronTrigger(**case)
        with pytest.raises(ValueError) as excinfo:
            scheduler.create_schedule(
                db,
                mlrun.api.schemas.AuthInfo(),
                "project",
                "schedule-name",
                schemas.ScheduleKinds.local_function,
                do_nothing,
                cron_trigger,
            )
        assert "Cron trigger too frequent. no more then one job" in str(excinfo.value)


@pytest.mark.asyncio
async def test_create_schedule_failure_already_exists(
    db: Session, scheduler: Scheduler
):
    cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
    )

    with pytest.raises(
        mlrun.errors.MLRunConflictError,
        match=rf"Conflict - Schedule already exists: {project}/{schedule_name}",
    ):
        scheduler.create_schedule(
            db,
            mlrun.api.schemas.AuthInfo(),
            project,
            schedule_name,
            schemas.ScheduleKinds.local_function,
            do_nothing,
            cron_trigger,
        )


@pytest.mark.asyncio
async def test_validate_cron_trigger_multi_checks(db: Session, scheduler: Scheduler):
    """
    _validate_cron_trigger runs 60 checks to be able to validate limit low as one minute.
    If we would run the check there one time it won't catch scenarios like:
    If the limit is 10 minutes and the cron trigger configured with minute=0-45 (which means every minute, for the
    first 45 minutes of every hour), and the check will occur at the 44 minute of some hour, the next run time
    will be one minute away, but the second next run time after it, will be at the next hour 0 minute. The delta
    between the two will be 15 minutes, more then 10 minutes so it will pass validation, although it actually runs
    every minute.
    """
    scheduler._min_allowed_interval = "10 minutes"
    cron_trigger = schemas.ScheduleCronTrigger(minute="0-45")
    now = datetime(
        year=2020,
        month=2,
        day=3,
        hour=4,
        minute=44,
        second=30,
        tzinfo=cron_trigger.timezone,
    )
    with pytest.raises(ValueError) as excinfo:
        scheduler._validate_cron_trigger(cron_trigger, now)
    assert "Cron trigger too frequent. no more then one job" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_schedule_datetime_fields_timezone(db: Session, scheduler: Scheduler):
    cron_trigger = schemas.ScheduleCronTrigger(minute="*/10")
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
    )
    schedule = scheduler.get_schedule(db, project, schedule_name)
    assert schedule.creation_time.tzinfo is not None
    assert schedule.next_run_time.tzinfo is not None

    schedules = scheduler.list_schedules(db, project)
    assert len(schedules.schedules) == 1
    assert schedules.schedules[0].creation_time.tzinfo is not None
    assert schedules.schedules[0].next_run_time.tzinfo is not None


@pytest.mark.asyncio
async def test_get_schedule(db: Session, scheduler: Scheduler):
    labels_1 = {
        "label1": "value1",
        "label2": "value2",
    }
    cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
        labels_1,
    )
    schedule = scheduler.get_schedule(db, project, schedule_name)

    # no next run time cause we put year=1999
    _assert_schedule(
        schedule,
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        cron_trigger,
        None,
        labels_1,
    )

    labels_2 = {
        "label3": "value3",
        "label4": "value4",
    }
    year = 2050
    cron_trigger_2 = schemas.ScheduleCronTrigger(year=year, timezone="utc")
    schedule_name_2 = "schedule-name-2"
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        schedule_name_2,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger_2,
        labels_2,
    )
    schedule_2 = scheduler.get_schedule(db, project, schedule_name_2)
    year_datetime = datetime(year=year, month=1, day=1, tzinfo=timezone.utc)
    _assert_schedule(
        schedule_2,
        project,
        schedule_name_2,
        schemas.ScheduleKinds.local_function,
        cron_trigger_2,
        year_datetime,
        labels_2,
    )

    schedules = scheduler.list_schedules(db)
    assert len(schedules.schedules) == 2
    _assert_schedule(
        schedules.schedules[0],
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        cron_trigger,
        None,
        labels_1,
    )
    _assert_schedule(
        schedules.schedules[1],
        project,
        schedule_name_2,
        schemas.ScheduleKinds.local_function,
        cron_trigger_2,
        year_datetime,
        labels_2,
    )

    schedules = scheduler.list_schedules(db, labels="label3=value3")
    assert len(schedules.schedules) == 1
    _assert_schedule(
        schedules.schedules[0],
        project,
        schedule_name_2,
        schemas.ScheduleKinds.local_function,
        cron_trigger_2,
        year_datetime,
        labels_2,
    )


@pytest.mark.asyncio
async def test_get_schedule_next_run_time_from_db(db: Session, scheduler: Scheduler):
    cron_trigger = schemas.ScheduleCronTrigger(minute="*/10")
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
    )
    chief_schedule = scheduler.get_schedule(db, project, schedule_name)
    assert chief_schedule.next_run_time is not None

    # simulating when running in worker
    mlrun.mlconf.httpdb.clusterization.role = (
        mlrun.api.schemas.ClusterizationRole.worker
    )
    worker_schedule = scheduler.get_schedule(db, project, schedule_name)
    assert worker_schedule.next_run_time is not None
    assert chief_schedule.next_run_time == worker_schedule.next_run_time


@pytest.mark.asyncio
async def test_get_schedule_failure_not_found(db: Session, scheduler: Scheduler):
    schedule_name = "schedule-name"
    project = config.default_project
    with pytest.raises(mlrun.errors.MLRunNotFoundError) as excinfo:
        scheduler.get_schedule(db, project, schedule_name)
    assert "Schedule not found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_list_schedules_name_filter(db: Session, scheduler: Scheduler):
    cases = [
        {"name": "some_prefix-mlrun", "should_find": True},
        {"name": "some_prefix-mlrun-some_suffix", "should_find": True},
        {"name": "mlrun-some_suffix", "should_find": True},
        {"name": "mlrun", "should_find": True},
        {"name": "MLRun", "should_find": True},
        {"name": "bla-MLRun-bla", "should_find": True},
        {"name": "mlun", "should_find": False},
        {"name": "mlurn", "should_find": False},
        {"name": "mluRn", "should_find": False},
    ]

    cron_trigger = schemas.ScheduleCronTrigger(minute="*/10")
    project = config.default_project
    expected_schedule_names = []
    for case in cases:
        name = case["name"]
        should_find = case["should_find"]
        scheduler.create_schedule(
            db,
            mlrun.api.schemas.AuthInfo(),
            project,
            name,
            schemas.ScheduleKinds.local_function,
            do_nothing,
            cron_trigger,
        )
        if should_find:
            expected_schedule_names.append(name)

    schedules = scheduler.list_schedules(db, project, "~mlrun")
    assert len(schedules.schedules) == len(expected_schedule_names)
    for schedule in schedules.schedules:
        assert schedule.name in expected_schedule_names
        expected_schedule_names.remove(schedule.name)


@pytest.mark.asyncio
async def test_list_schedules_from_scheduler(db: Session, scheduler: Scheduler):
    project_1 = "project-1"
    project_1_number_of_schedules = 5
    for index in range(project_1_number_of_schedules):
        schedule_name = f"schedule-name-{index}"
        _create_do_nothing_schedule(db, scheduler, project_1, schedule_name)
    project_2 = "project-2"
    project_2_number_of_schedules = 2
    for index in range(project_2_number_of_schedules):
        schedule_name = f"schedule-name-{index}"
        _create_do_nothing_schedule(db, scheduler, project_2, schedule_name)
    assert (
        len(scheduler._list_schedules_from_scheduler(project_1))
        == project_1_number_of_schedules
    )
    assert (
        len(scheduler._list_schedules_from_scheduler(project_2))
        == project_2_number_of_schedules
    )


@pytest.mark.asyncio
async def test_delete_schedule(db: Session, scheduler: Scheduler):
    cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
    )

    schedules = scheduler.list_schedules(db)
    assert len(schedules.schedules) == 1

    scheduler.delete_schedule(db, project, schedule_name)

    schedules = scheduler.list_schedules(db)
    assert len(schedules.schedules) == 0

    # verify another delete pass successfully
    scheduler.delete_schedule(db, project, schedule_name)


@pytest.mark.asyncio
async def test_delete_schedules(db: Session, scheduler: Scheduler):
    project = config.default_project
    number_of_schedules = 5
    for index in range(number_of_schedules):
        schedule_name = f"schedule-name-{index}"
        _create_do_nothing_schedule(db, scheduler, project, schedule_name)

    schedules = scheduler.list_schedules(db)
    assert len(schedules.schedules) == number_of_schedules

    scheduler.delete_schedules(db, project)

    schedules = scheduler.list_schedules(db)
    assert len(schedules.schedules) == 0

    # verify another delete pass successfully
    scheduler.delete_schedules(db, project)


@pytest.mark.asyncio
async def test_rescheduling(db: Session, scheduler: Scheduler):
    global call_counter
    call_counter = 0

    expected_call_counter = 2
    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=expected_call_counter, seconds_interval=1
    )
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1", start_date=start_date, end_date=end_date
    )
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        bump_counter,
        cron_trigger,
    )

    # wait so one run will complete
    time_to_sleep = (start_date - datetime.now()).total_seconds() + 1
    await asyncio.sleep(time_to_sleep)

    # stop the scheduler and assert indeed only one call happened
    await scheduler.stop()
    assert call_counter == 1

    # start the scheduler and and assert another run
    await scheduler.start(db)
    await asyncio.sleep(1 + schedule_end_time_margin)
    assert call_counter == 2


@pytest.mark.asyncio
async def test_rescheduling_secrets_storing(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    name = "schedule-name"
    project = config.default_project
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(db, project)
    username = "some-username"
    access_key = "some-user-access-key"
    cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(username=username, access_key=access_key),
        project,
        name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )

    jobs = scheduler._list_schedules_from_scheduler(project)
    assert jobs[0].args[5].access_key == access_key
    assert jobs[0].args[5].username == username
    k8s_secrets_mock.assert_auth_secret(
        k8s_secrets_mock.get_auth_secret_name(username, access_key),
        username,
        access_key,
    )

    await scheduler.stop()

    jobs = scheduler._list_schedules_from_scheduler(project)
    assert jobs == []

    await scheduler.start(db)
    jobs = scheduler._list_schedules_from_scheduler(project)
    assert jobs[0].args[5].username == username
    assert jobs[0].args[5].access_key == access_key


@pytest.mark.asyncio
async def test_schedule_crud_secrets_handling(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    for schedule_name in ["valid-secret-key", "invalid/secret/key"]:
        project = config.default_project
        scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
            db, project
        )
        access_key = "some-user-access-key"
        username = "some-username"
        cron_trigger = schemas.ScheduleCronTrigger(year="1999")
        scheduler.create_schedule(
            db,
            mlrun.api.schemas.AuthInfo(username=username, access_key=access_key),
            project,
            schedule_name,
            schemas.ScheduleKinds.job,
            scheduled_object,
            cron_trigger,
        )
        _assert_schedule_auth_secrets(
            k8s_secrets_mock.get_auth_secret_name(username, access_key),
            username,
            access_key,
        )
        _assert_schedule_get_and_list_credentials_enrichment(
            db, scheduler, project, schedule_name, access_key, username
        )

        username = "new-username"
        access_key = "new-access-key"
        # update labels
        scheduler.update_schedule(
            db,
            mlrun.api.schemas.AuthInfo(username=username, access_key=access_key),
            project,
            schedule_name,
            labels={"label-key": "label-value"},
        )

        _assert_schedule_auth_secrets(
            k8s_secrets_mock.get_auth_secret_name(username, access_key),
            username,
            access_key,
        )
        _assert_schedule_get_and_list_credentials_enrichment(
            db, scheduler, project, schedule_name, access_key, username
        )

        # delete schedule
        scheduler.delete_schedule(
            db,
            project,
            schedule_name,
        )


@pytest.mark.asyncio
async def test_schedule_access_key_generation(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    project = config.default_project
    schedule_name = "schedule-name"
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(db, project)
    cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    access_key = "generated-access-key"
    mlrun.api.utils.auth.verifier.AuthVerifier().get_or_create_access_key = (
        unittest.mock.Mock(return_value=access_key)
    )
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        schedule_name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().get_or_create_access_key.assert_called_once()
    _assert_schedule_auth_secrets(
        k8s_secrets_mock.get_auth_secret_name("", access_key), "", access_key
    )

    access_key = "generated-access-key-2"
    mlrun.api.utils.auth.verifier.AuthVerifier().get_or_create_access_key = (
        unittest.mock.Mock(return_value=access_key)
    )
    scheduler.update_schedule(
        db,
        mlrun.api.schemas.AuthInfo(
            access_key=mlrun.model.Credentials.generate_access_key
        ),
        project,
        schedule_name,
        labels={"label-key": "label-value"},
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().get_or_create_access_key.assert_called_once()
    _assert_schedule_auth_secrets(
        k8s_secrets_mock.get_auth_secret_name("", access_key), "", access_key
    )


@pytest.mark.asyncio
async def test_schedule_access_key_reference_handling(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    project = config.default_project
    schedule_name = "schedule-name"
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(db, project)
    cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    username = "some-user-name"
    access_key = "some-access-key"

    secret_ref = (
        mlrun.model.Credentials.secret_reference_prefix
        + k8s_secrets_mock.store_auth_secret(username, access_key)
    )
    auth_info = mlrun.api.schemas.AuthInfo()
    auth_info.access_key = secret_ref

    scheduler.create_schedule(
        db,
        auth_info,
        project,
        schedule_name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
        labels={"label1": "value1", "label2": "value2"},
    )

    _assert_schedule_get_and_list_credentials_enrichment(
        db, scheduler, project, schedule_name, access_key, username
    )


@pytest.mark.asyncio
async def test_schedule_convert_from_old_credentials_to_new(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    # Verify that objects created with the old methodology of storing secrets are converted properly when the
    # scheduler is reloaded, and new auth secrets are generated for them.
    project = config.default_project
    schedule_name = "schedule-name"
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(db, project)
    cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    username = "some-user-name"
    access_key = "some-access-key"

    # Creating without auth info and without setting is_jobs_auth_required, since we will create the secrets later
    # to simulate an old schedule.
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        schedule_name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
        labels={"label1": "value1", "label2": "value2"},
    )

    auth_info = mlrun.api.schemas.AuthInfo(username=username, access_key=access_key)
    mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    scheduler._store_schedule_secrets(auth_info, project, schedule_name)
    _assert_schedule_secrets(scheduler, project, schedule_name, username, access_key)

    # now reload the schedules, to trigger conversion.

    await scheduler.stop()

    jobs = scheduler._list_schedules_from_scheduler(project)
    assert jobs == []

    await scheduler.start(db)
    _assert_schedule_auth_secrets(
        k8s_secrets_mock.get_auth_secret_name(username, access_key),
        username,
        access_key,
    )

    jobs = scheduler._list_schedules_from_scheduler(project)
    assert jobs[0].args[5].username == username
    assert jobs[0].args[5].access_key == access_key
    _assert_schedule_get_and_list_credentials_enrichment(
        db, scheduler, project, schedule_name, access_key, username
    )
    _assert_schedule_secrets(scheduler, project, schedule_name, None, None)


@pytest.mark.asyncio
async def test_update_schedule(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    labels_1 = {
        "label1": "value1",
        "label2": "value2",
    }
    labels_2 = {
        "label3": "value3",
        "label4": "value4",
    }
    inactive_cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "schedule-name"
    project_name = config.default_project
    mlrun.new_project(project_name, save=False)

    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project_name,
        schedule_name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        inactive_cron_trigger,
        labels=labels_1,
    )

    schedule = scheduler.get_schedule(db, project_name, schedule_name)

    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        labels_1,
    )

    # update labels
    scheduler.update_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project_name,
        schedule_name,
        labels=labels_2,
    )
    schedule = scheduler.get_schedule(db, project_name, schedule_name)

    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        labels_2,
    )

    # update nothing
    scheduler.update_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project_name,
        schedule_name,
    )
    schedule = scheduler.get_schedule(db, project_name, schedule_name)

    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        labels_2,
    )

    # update labels to empty dict
    scheduler.update_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project_name,
        schedule_name,
        labels={},
    )
    schedule = scheduler.get_schedule(db, project_name, schedule_name)

    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        {},
    )

    # update it so it runs
    expected_call_counter = 1
    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=expected_call_counter, seconds_interval=1
    )
    # this way we're leaving ourselves one second to create the schedule preventing transient test failure
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1",
        start_date=start_date,
        end_date=end_date,
    )
    scheduler.update_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project_name,
        schedule_name,
        cron_trigger=cron_trigger,
    )
    schedule = scheduler.get_schedule(db, project_name, schedule_name)

    next_run_time = datetime(
        year=end_date.year,
        month=end_date.month,
        day=end_date.day,
        hour=end_date.hour,
        minute=end_date.minute,
        second=end_date.second,
        tzinfo=tzlocal(),
    )

    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        schemas.ScheduleKinds.job,
        cron_trigger,
        next_run_time,
        {},
    )
    time_to_sleep = (
        end_date - datetime.now()
    ).total_seconds() + schedule_end_time_margin

    await asyncio.sleep(time_to_sleep)
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 1
    assert runs[0]["status"]["state"] == RunStates.completed


@pytest.mark.asyncio
async def test_update_schedule_failure_not_found_in_db(
    db: Session, scheduler: Scheduler
):
    schedule_name = "schedule-name"
    project = config.default_project
    with pytest.raises(mlrun.errors.MLRunNotFoundError) as excinfo:
        scheduler.update_schedule(
            db, mlrun.api.schemas.AuthInfo(), project, schedule_name
        )
    assert "Schedule not found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_update_schedule_failure_not_found_in_scheduler(
    db: Session, scheduler: Scheduler
):
    schedule_name = "schedule-name"
    project_name = config.default_project
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name
    )

    # create the schedule only in the db
    inactive_cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    get_db().create_schedule(
        db,
        project_name,
        schedule_name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        inactive_cron_trigger,
        1,
    )

    # update schedule should fail since the schedule job was not created in the scheduler
    with pytest.raises(mlrun.errors.MLRunNotFoundError) as excinfo:
        scheduler.update_schedule(
            db, mlrun.api.schemas.AuthInfo(), project_name, schedule_name
        )
    job_id = scheduler._resolve_job_id(project_name, schedule_name)
    assert (
        f"Schedule job with id {job_id} not found in scheduler. Reload schedules is required."
        in str(excinfo.value)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # The function waits 2 seconds and the schedule runs every second for 4 seconds. So:
    # For 1 concurrent job, the second and fourth jobs should be skipped resulting in 2 runs.
    # For 2 concurrent jobs, the third job should be skipped resulting in 3 runs.
    # For 3 concurrent jobs, no job should be skipped resulting in 4 runs.
    "concurrency_limit,run_amount",
    [(1, 2), (2, 3), (3, 4)],
)
@pytest.mark.parametrize(
    "schedule_kind", [schemas.ScheduleKinds.job, schemas.ScheduleKinds.local_function]
)
async def test_schedule_job_concurrency_limit(
    db: Session,
    scheduler: Scheduler,
    concurrency_limit: int,
    run_amount: int,
    schedule_kind: schemas.ScheduleKinds,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    global call_counter
    call_counter = 0

    now = datetime.now()
    now_plus_1_seconds = now + timedelta(seconds=1)
    now_plus_5_seconds = now + timedelta(seconds=5)
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1", start_date=now_plus_1_seconds, end_date=now_plus_5_seconds
    )
    schedule_name = "schedule-name"
    project_name = config.default_project
    mlrun.new_project(project_name, save=False)

    scheduled_object = (
        _create_mlrun_function_and_matching_scheduled_object(
            db, project_name, handler="sleep_two_seconds"
        )
        if schedule_kind == schemas.ScheduleKinds.job
        else bump_counter_and_wait
    )

    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0

    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project_name,
        schedule_name,
        schedule_kind,
        scheduled_object,
        cron_trigger,
        concurrency_limit=concurrency_limit,
    )

    random_sleep_time = random.randint(1, 5)
    await asyncio.sleep(random_sleep_time)
    schedule = scheduler.get_schedule(
        db,
        project_name,
        schedule_name,
    )

    # scrub the microseconds to reduce noise
    now = datetime.now(timezone.utc).replace(microsecond=0)
    if random_sleep_time == 5:
        # next run time may be none if the job was completed (i.e. end date was reached)
        assert schedule.next_run_time is None
    elif random_sleep_time == 4 and schedule.next_run_time is not None:
        assert schedule.next_run_time >= now
    else:
        assert schedule.next_run_time >= now

    # wait so all runs will complete
    await asyncio.sleep(7 - random_sleep_time)
    if schedule_kind == schemas.ScheduleKinds.job:
        runs = get_db().list_runs(db, project=project_name)
        assert len(runs) == run_amount
    else:
        assert call_counter == run_amount


@pytest.mark.asyncio
async def test_schedule_job_next_run_time(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    """
    This test checks that the next run time is updated after a schedule was skipped due to concurrency limit.
    It creates a schedule that runs every second for 4 seconds with concurrency limit of 1.
    The run takes 2 seconds to complete so the function should be triggered twice in that time frame.
    While the 1st run is still running, manually invoke the schedule (should fail due to concurrency limit)
    and check that the next run time is updated.
    """
    now_plus_4_seconds = datetime.now() + timedelta(seconds=4)
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1", end_date=now_plus_4_seconds
    )
    schedule_name = "schedule-name"
    project_name = config.default_project
    mlrun.new_project(project_name, save=False)

    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name, handler="sleep_two_seconds"
    )

    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0

    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project_name,
        schedule_name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
        concurrency_limit=1,
    )

    await asyncio.sleep(1)
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 1

    # invocation should fail due to concurrency limit
    # the next run time should be updated to the next second after the invocation failure
    schedule_invocation_timestamp = datetime.now(timezone.utc)
    await scheduler.invoke_schedule(
        db, mlrun.api.schemas.AuthInfo(), project_name, schedule_name
    )

    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 1

    # assert next run time was updated
    schedule = scheduler.get_schedule(
        db,
        project_name,
        schedule_name,
    )
    assert schedule.next_run_time > schedule_invocation_timestamp

    # wait so all runs will complete
    await asyncio.sleep(5)
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 2


def _assert_schedule_get_and_list_credentials_enrichment(
    db: Session,
    scheduler: Scheduler,
    project: str,
    schedule_name: str,
    expected_access_key: str,
    expected_username: str,
):
    schedule = scheduler.get_schedule(
        db,
        project,
        schedule_name,
        include_credentials=True,
    )

    secret_name = tests.api.conftest.K8sSecretsMock.get_auth_secret_name(
        expected_username, expected_access_key
    )
    secret_ref = mlrun.model.Credentials.secret_reference_prefix + secret_name

    assert schedule.labels[scheduler._db_record_auth_label] == secret_name
    assert schedule.credentials.access_key == secret_ref
    schedules = scheduler.list_schedules(
        db, project, schedule_name, include_credentials=True
    )
    assert schedules.schedules[0].credentials.access_key == secret_ref

    jobs = scheduler._list_schedules_from_scheduler(project)
    assert jobs[0].args[5].access_key == expected_access_key
    assert jobs[0].args[5].username == expected_username


def _assert_schedule_auth_secrets(
    secret_name: str,
    expected_username: str,
    expected_access_key: str,
):
    auth_data = mlrun.api.crud.Secrets().read_auth_secret(secret_name)
    assert expected_username == auth_data.username
    assert expected_access_key == auth_data.access_key


def _assert_schedule_secrets(
    scheduler: Scheduler,
    project: str,
    schedule_name: str,
    expected_username: str,
    expected_access_key: str,
):
    access_key_secret_key = mlrun.api.crud.Secrets().generate_client_project_secret_key(
        mlrun.api.crud.SecretsClientType.schedules,
        schedule_name,
        scheduler._secret_access_key_subtype,
    )
    username_secret_key = mlrun.api.crud.Secrets().generate_client_project_secret_key(
        mlrun.api.crud.SecretsClientType.schedules,
        schedule_name,
        scheduler._secret_username_subtype,
    )
    key_map_secret_key = (
        mlrun.api.crud.Secrets().generate_client_key_map_project_secret_key(
            mlrun.api.crud.SecretsClientType.schedules
        )
    )
    secret_value = mlrun.api.crud.Secrets().get_project_secret(
        project,
        scheduler._secrets_provider,
        access_key_secret_key,
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )
    assert secret_value == expected_access_key
    secret_value = mlrun.api.crud.Secrets().get_project_secret(
        project,
        scheduler._secrets_provider,
        username_secret_key,
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )
    assert secret_value == expected_username


def _assert_schedule(
    schedule: schemas.ScheduleOutput,
    project,
    name,
    kind,
    cron_trigger,
    next_run_time,
    labels,
):
    assert schedule.name == name
    assert schedule.project == project
    assert schedule.kind == kind
    assert schedule.next_run_time == next_run_time
    assert schedule.cron_trigger == cron_trigger
    assert schedule.creation_time is not None
    assert DeepDiff(schedule.labels, labels, ignore_order=True) == {}


def _create_do_nothing_schedule(
    db: Session, scheduler: Scheduler, project: str, name: str
):
    cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    scheduler.create_schedule(
        db,
        mlrun.api.schemas.AuthInfo(),
        project,
        name,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
    )


def _create_mlrun_function_and_matching_scheduled_object(
    db: Session, project: str, handler: str = "do_nothing"
):
    function_name = "my-function"
    code_path = pathlib.Path(__file__).absolute().parent / "function.py"
    function = mlrun.code_to_function(
        name=function_name, kind="local", filename=str(code_path)
    )
    function.spec.command = f"{str(code_path)}"
    hash_key = get_db().store_function(
        db, function.to_dict(), function_name, project, versioned=True
    )
    scheduled_object = {
        "task": {
            "spec": {
                "function": f"{project}/{function_name}@{hash_key}",
                "handler": handler,
            },
            "metadata": {"name": "my-task", "project": f"{project}"},
        }
    }
    return scheduled_object


def _get_start_and_end_time_for_scheduled_trigger(
    number_of_jobs: int, seconds_interval: int
):
    """
    The scheduler executes the job on round seconds (when microsecond == 0)
    Therefore if the start time will be a round second - let's say 12:08:06.000000 and the end time 12:08:07.000000
    it means two executions will happen - at 06 and 07 second.
    This is obviously very rare (since the times are based on datetime.now()) - usually the start time
    will be something like 12:08:06.100000 then the end time will be 12:08:07.10000 - meaning there will be only
     one execution on the 07 second.
    So instead of conditioning every assertion we're doing on whether the start time was a round second,
     we simply make sure it's not a round second.
    """
    now = datetime.now()
    if now.microsecond == 0:
        now = now + timedelta(seconds=1, milliseconds=1)
    start_date = now + timedelta(seconds=1)
    end_date = now + timedelta(seconds=1 + number_of_jobs * seconds_interval)
    return start_date, end_date
