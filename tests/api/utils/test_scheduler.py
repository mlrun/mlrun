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
import asyncio
import pathlib
import random
import time
import typing
import unittest.mock
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from deepdiff import DeepDiff
from sqlalchemy.orm import Session

import mlrun
import mlrun.common.schemas
import mlrun.errors
import mlrun.launcher.factory
import server.api.crud
import server.api.utils.auth
import server.api.utils.auth.verifier
import server.api.utils.helpers
import server.api.utils.singletons.project_member
import tests.api.conftest
from mlrun.common.runtimes.constants import RunStates
from mlrun.config import config
from mlrun.utils import logger
from server.api.utils.scheduler import Scheduler
from server.api.utils.singletons.db import get_db


@pytest_asyncio.fixture()
async def scheduler(db: Session) -> typing.AsyncIterator[Scheduler]:
    logger.info("Creating scheduler")
    config.httpdb.scheduling.min_allowed_interval = "0"
    config.httpdb.jobs.allow_local_run = True

    # The scheduler tests are subject to running local functions.
    # Since running local functions is not supported in the API, we need to run as client.
    mlrun.config._is_running_as_api = False

    scheduler = Scheduler()
    await scheduler.start(db)
    server.api.utils.singletons.project_member.initialize_project_member()
    yield scheduler
    logger.info("Stopping scheduler")
    await scheduler.stop()


call_counter: int = 0

# TODO: The margin will need to rise for each additional CPU-consuming operation added along the flow,
#  we need to consider how to decouple in the future
schedule_end_time_margin = 0.7


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


def create_project(
    db: Session, project_name: str = None
) -> mlrun.common.schemas.Project:
    """API tests use sql db, so we need to create the project with its schema"""
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(
            name=project_name or config.default_project
        )
    )
    server.api.crud.Projects().create_project(db, project)
    return project


@pytest.mark.asyncio
async def test_not_skipping_delayed_schedules(db: Session, scheduler: Scheduler):
    global call_counter
    call_counter = 0
    expected_call_counter = 1

    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=expected_call_counter, seconds_interval=1
    )
    # this way we're leaving ourselves one second to create the schedule preventing transient test failure
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(
        second="*/1", start_date=start_date, end_date=end_date
    )
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.local_function,
        bump_counter,
        cron_trigger,
    )
    # purposely doing time.sleep to block the reactor to ensure a job is still scheduled although its planned
    # execution time passed
    time.sleep(2 + expected_call_counter)
    await asyncio.sleep(1)
    assert call_counter == expected_call_counter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store",
    [False, True],
)
async def test_create_schedule(db: Session, scheduler: Scheduler, store: bool):
    global call_counter
    call_counter = 0

    expected_call_counter = 5
    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=5, seconds_interval=1
    )
    # this way we're leaving ourselves one second to create the schedule preventing transient test failure
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(
        second="*/1", start_date=start_date, end_date=end_date
    )
    schedule_name = "schedule-name"
    project = config.default_project
    if store:
        scheduler.store_schedule(
            db_session=db,
            auth_info=mlrun.common.schemas.AuthInfo(),
            project=project,
            name=schedule_name,
            kind=mlrun.common.schemas.ScheduleKinds.local_function,
            scheduled_object=bump_counter,
            cron_trigger=cron_trigger,
        )

    else:
        scheduler.create_schedule(
            db_session=db,
            auth_info=mlrun.common.schemas.AuthInfo(),
            project=project,
            name=schedule_name,
            kind=mlrun.common.schemas.ScheduleKinds.local_function,
            scheduled_object=bump_counter,
            cron_trigger=cron_trigger,
        )

    # The trigger is defined with `second="*/1"` meaning it runs on round seconds,
    # but executing the actual functional code - bumping the counter - happens a few microseconds afterwards.
    # To avoid transient errors on slow systems, we add extra margin.
    time_to_sleep = (
        end_date - mlrun.utils.now_date()
    ).total_seconds() + schedule_end_time_margin

    await asyncio.sleep(time_to_sleep)
    assert call_counter == expected_call_counter


@pytest.mark.asyncio
async def test_invoke_schedule(
    db: Session,
    client: tests.api.conftest.TestClient,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year=1999)
    schedule_name = "schedule-name"
    project_name = config.default_project
    create_project(db, project_name)
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0
    response_1 = await scheduler.invoke_schedule(
        db, mlrun.common.schemas.AuthInfo(), project_name, schedule_name
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 1
    response_2 = await scheduler.invoke_schedule(
        db, mlrun.common.schemas.AuthInfo(), project_name, schedule_name
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
# ML-4902
async def test_get_schedule_last_run_deleted(
    db: Session,
    client: tests.api.conftest.TestClient,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year=1999)
    schedule_name = "schedule-name"
    project_name = config.default_project
    create_project(db, project_name)
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name
    )
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )
    await scheduler.invoke_schedule(
        db, mlrun.common.schemas.AuthInfo(), project_name, schedule_name
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 1

    run_uid = runs[0]["metadata"]["uid"]
    schedule = scheduler.get_schedule(
        db, project_name, schedule_name, include_last_run=True
    )

    assert schedule.last_run is not None
    assert schedule.last_run["metadata"]["uid"] == run_uid
    assert schedule.last_run["metadata"]["project"] == project_name

    # delete the last run for the schedule, ensure we can still get the schedule without failing
    get_db().del_run(db, uid=run_uid, project=project_name)
    schedule = scheduler.get_schedule(
        db, project_name, schedule_name, include_last_run=True
    )

    assert schedule.last_run_uri is None
    assert schedule.last_run == {}


@pytest.mark.asyncio
async def test_create_schedule_mlrun_function(
    db: Session,
    client: tests.api.conftest.TestClient,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    project_name = config.default_project
    create_project(db, project_name)

    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0

    expected_call_counter = 1
    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=expected_call_counter, seconds_interval=1
    )
    # this way we're leaving ourselves one second to create the schedule preventing transient test failure
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(
        second="*/1", start_date=start_date, end_date=end_date
    )
    schedule_name = "schedule-name"
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )
    time_to_sleep = (
        end_date - mlrun.utils.now_date()
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
        cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(**case)
        scheduler.create_schedule(
            db,
            mlrun.common.schemas.AuthInfo(),
            "project",
            f"schedule-name-{index}",
            mlrun.common.schemas.ScheduleKinds.local_function,
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
    create_project(db, project_name)

    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name
    )

    expected_call_counter = 3
    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=expected_call_counter, seconds_interval=1
    )
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(
        second="*/1", start_date=start_date, end_date=end_date
    )
    # we're before upgrade so create a schedule with empty auth info
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project_name,
        name,
        mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )
    # stop scheduler, reconfigure to store credentials and start again (upgrade)
    await scheduler.stop()
    server.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    await scheduler.start(db)

    # at this point the schedule is inside the scheduler without auth_info, so the first trigger should try to generate
    # auth info, mock the functions for this
    username = "some-username"
    access_key = "some-access_key"
    server.api.utils.singletons.project_member.get_project_member().get_project_owner = unittest.mock.Mock(
        return_value=mlrun.common.schemas.ProjectOwner(
            username=username, access_key=access_key
        )
    )
    time_to_sleep = (
        end_date - mlrun.utils.now_date()
    ).total_seconds() + schedule_end_time_margin

    await asyncio.sleep(time_to_sleep)
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 3
    assert (
        server.api.utils.singletons.project_member.get_project_member().get_project_owner.call_count
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
        cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(**case)
        with pytest.raises(ValueError) as excinfo:
            scheduler.create_schedule(
                db,
                mlrun.common.schemas.AuthInfo(),
                "project",
                "schedule-name",
                mlrun.common.schemas.ScheduleKinds.local_function,
                do_nothing,
                cron_trigger,
            )
        assert "Cron trigger too frequent. no more than one job" in str(excinfo.value)


@pytest.mark.asyncio
async def test_create_schedule_failure_already_exists(
    db: Session, scheduler: Scheduler
):
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
    )

    with pytest.raises(
        mlrun.errors.MLRunConflictError,
        match=rf"Conflict - at least one of the objects already exists: {project}/{schedule_name}",
    ):
        scheduler.create_schedule(
            db,
            mlrun.common.schemas.AuthInfo(),
            project,
            schedule_name,
            mlrun.common.schemas.ScheduleKinds.local_function,
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
    between the two will be 15 minutes, more than 10 minutes so it will pass validation, although it actually runs
    every minute.
    """
    scheduler._min_allowed_interval = "10 minutes"
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(minute="0-45")
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
    assert "Cron trigger too frequent. no more than one job" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_schedule_datetime_fields_timezone(db: Session, scheduler: Scheduler):
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(minute="*/10")
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.local_function,
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
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.local_function,
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
        mlrun.common.schemas.ScheduleKinds.local_function,
        cron_trigger,
        None,
        labels_1,
        config.httpdb.scheduling.default_concurrency_limit,
    )

    labels_2 = {
        "label3": "value3",
        "label4": "value4",
    }
    year = 2050
    cron_trigger_2 = mlrun.common.schemas.ScheduleCronTrigger(year=year, timezone="utc")
    schedule_name_2 = "schedule-name-2"
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project,
        schedule_name_2,
        mlrun.common.schemas.ScheduleKinds.local_function,
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
        mlrun.common.schemas.ScheduleKinds.local_function,
        cron_trigger_2,
        year_datetime,
        labels_2,
        config.httpdb.scheduling.default_concurrency_limit,
    )

    schedules = scheduler.list_schedules(db)
    assert len(schedules.schedules) == 2
    _assert_schedule(
        schedules.schedules[0],
        project,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.local_function,
        cron_trigger,
        None,
        labels_1,
        config.httpdb.scheduling.default_concurrency_limit,
    )
    _assert_schedule(
        schedules.schedules[1],
        project,
        schedule_name_2,
        mlrun.common.schemas.ScheduleKinds.local_function,
        cron_trigger_2,
        year_datetime,
        labels_2,
        config.httpdb.scheduling.default_concurrency_limit,
    )

    schedules = scheduler.list_schedules(db, labels="label3=value3")
    assert len(schedules.schedules) == 1
    _assert_schedule(
        schedules.schedules[0],
        project,
        schedule_name_2,
        mlrun.common.schemas.ScheduleKinds.local_function,
        cron_trigger_2,
        year_datetime,
        labels_2,
        config.httpdb.scheduling.default_concurrency_limit,
    )


@pytest.mark.asyncio
async def test_get_schedule_next_run_time_from_db(db: Session, scheduler: Scheduler):
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(minute="*/10")
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
    )
    chief_schedule = scheduler.get_schedule(db, project, schedule_name)
    assert chief_schedule.next_run_time is not None

    # simulating when running in worker
    mlrun.mlconf.httpdb.clusterization.role = (
        mlrun.common.schemas.ClusterizationRole.worker
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

    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(minute="*/10")
    project = config.default_project
    expected_schedule_names = []
    for case in cases:
        name = case["name"]
        should_find = case["should_find"]
        scheduler.create_schedule(
            db,
            mlrun.common.schemas.AuthInfo(),
            project,
            name,
            mlrun.common.schemas.ScheduleKinds.local_function,
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
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.local_function,
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
    """
    Test flow:
        1. Create a schedule triggered every second
        2. Wait for one run to complete
        3. Stop the scheduler
        4. Start the scheduler - schedule should be reloaded
        5. Wait for another run to complete
    """
    global call_counter
    call_counter = 0

    # we expect 3 calls but assert 2 to avoid edge cases where the schedule was reloaded in the same second
    # as the end date and therefore doesn't trigger another run
    expected_call_counter = 3
    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=expected_call_counter, seconds_interval=1
    )
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(
        second="*/1", start_date=start_date, end_date=end_date
    )
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.local_function,
        bump_counter,
        cron_trigger,
    )

    # wait so one run will complete
    time_to_sleep = (start_date - mlrun.utils.now_date()).total_seconds() + 1
    await asyncio.sleep(time_to_sleep)

    # stop the scheduler and assert indeed only one call happened
    await scheduler.stop()
    assert call_counter == 1

    # start the scheduler and assert another run
    await scheduler.start(db)
    await asyncio.sleep(1 + schedule_end_time_margin)
    assert call_counter >= 2


@pytest.mark.asyncio
async def test_rescheduling_secrets_storing(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    server.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    name = "schedule-name"
    project = config.default_project
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(db, project)
    username = "some-username"
    access_key = "some-user-access-key"
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year="1999")
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(username=username, access_key=access_key),
        project,
        name,
        mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )

    jobs = scheduler._list_schedules_from_scheduler(project)
    assert jobs[0].args[5].access_key == access_key
    assert jobs[0].args[5].username == username
    k8s_secrets_mock.assert_auth_secret(
        k8s_secrets_mock.resolve_auth_secret_name(username, access_key),
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
    server.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    for schedule_name in ["valid-secret-key", "invalid/secret/key"]:
        project = config.default_project
        scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
            db, project
        )
        access_key = "some-user-access-key"
        username = "some-username"
        cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year="1999")
        scheduler.create_schedule(
            db,
            mlrun.common.schemas.AuthInfo(username=username, access_key=access_key),
            project,
            schedule_name,
            mlrun.common.schemas.ScheduleKinds.job,
            scheduled_object,
            cron_trigger,
        )
        _assert_schedule_auth_secrets(
            k8s_secrets_mock.resolve_auth_secret_name(username, access_key),
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
            mlrun.common.schemas.AuthInfo(username=username, access_key=access_key),
            project,
            schedule_name,
            labels={"label-key": "label-value"},
        )

        _assert_schedule_auth_secrets(
            k8s_secrets_mock.resolve_auth_secret_name(username, access_key),
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
    server.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    project = config.default_project
    schedule_name = "schedule-name"
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(db, project)
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year="1999")
    access_key = "generated-access-key"
    get_or_create_access_key_mock = unittest.mock.Mock(return_value=access_key)
    server.api.utils.auth.verifier.AuthVerifier().get_or_create_access_key = (
        get_or_create_access_key_mock
    )
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )
    get_or_create_access_key_mock.assert_called_once()
    _assert_schedule_auth_secrets(
        k8s_secrets_mock.resolve_auth_secret_name("", access_key), "", access_key
    )

    access_key = "generated-access-key-2"
    get_or_create_access_key_mock = unittest.mock.Mock(return_value=access_key)
    server.api.utils.auth.verifier.AuthVerifier().get_or_create_access_key = (
        get_or_create_access_key_mock
    )
    scheduler.update_schedule(
        db,
        mlrun.common.schemas.AuthInfo(
            access_key=mlrun.model.Credentials.generate_access_key
        ),
        project,
        schedule_name,
        labels={"label-key": "label-value"},
    )
    get_or_create_access_key_mock.assert_called_once()
    _assert_schedule_auth_secrets(
        k8s_secrets_mock.resolve_auth_secret_name("", access_key), "", access_key
    )


@pytest.mark.asyncio
async def test_schedule_access_key_reference_handling(
    db: Session,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    server.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    project = config.default_project
    schedule_name = "schedule-name"
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(db, project)
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year="1999")
    username = "some-user-name"
    access_key = "some-access-key"

    mocked_secret_ref, _ = k8s_secrets_mock.store_auth_secret(username, access_key)
    secret_ref = mlrun.model.Credentials.secret_reference_prefix + mocked_secret_ref
    auth_info = mlrun.common.schemas.AuthInfo()
    auth_info.access_key = secret_ref

    scheduler.create_schedule(
        db,
        auth_info,
        project,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
        labels={"label1": "value1", "label2": "value2"},
    )

    _assert_schedule_get_and_list_credentials_enrichment(
        db, scheduler, project, schedule_name, access_key, username
    )


@unittest.mock.patch.object(
    Scheduler, "_store_schedule_secrets_using_auth_secret", return_value="auth-secret"
)
@pytest.mark.asyncio
async def test_update_schedule(
    mock_store_schedule_secrets_using_auth_secret,
    db: Session,
    client: tests.api.conftest.TestClient,
    scheduler: Scheduler,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    labels_1 = {
        "label1": "value1",
        "label2": "value2",
    }
    expected_labels_1 = labels_1.copy()
    expected_labels_1.update({"mlrun-auth-key": "auth-secret"})
    labels_2 = {
        "label3": "value3",
        "label4": "value4",
    }
    expected_labels_2 = labels_2.copy()
    expected_labels_2.update({"mlrun-auth-key": "auth-secret"})
    inactive_cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "schedule-name"
    project_name = config.default_project
    create_project(db, project_name)

    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object,
        inactive_cron_trigger,
        labels=labels_1.copy(),
    )

    schedule = scheduler.get_schedule(db, project_name, schedule_name)

    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        expected_labels_1,
        config.httpdb.scheduling.default_concurrency_limit,
    )

    # update labels
    scheduler.update_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project_name,
        schedule_name,
        labels=labels_2,
    )
    schedule = scheduler.get_schedule(db, project_name, schedule_name)

    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        expected_labels_2,
        config.httpdb.scheduling.default_concurrency_limit,
    )

    # update nothing
    scheduler.update_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project_name,
        schedule_name,
    )
    schedule = scheduler.get_schedule(db, project_name, schedule_name)

    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        expected_labels_2,
        config.httpdb.scheduling.default_concurrency_limit,
    )

    # update labels to empty dict
    scheduler.update_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project_name,
        schedule_name,
        labels={},
    )
    schedule = scheduler.get_schedule(db, project_name, schedule_name)

    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        {"mlrun-auth-key": "auth-secret"},
        config.httpdb.scheduling.default_concurrency_limit,
    )

    # update it so it runs
    expected_call_counter = 1
    start_date, end_date = _get_start_and_end_time_for_scheduled_trigger(
        number_of_jobs=expected_call_counter, seconds_interval=1
    )
    # this way we're leaving ourselves one second to create the schedule preventing transient test failure
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(
        second="*/1",
        start_date=start_date,
        end_date=end_date,
    )
    scheduler.update_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
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
        tzinfo=timezone.utc,
    )

    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        cron_trigger,
        next_run_time,
        {"mlrun-auth-key": "auth-secret"},
        config.httpdb.scheduling.default_concurrency_limit,
    )
    time_to_sleep = (
        end_date - mlrun.utils.now_date()
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
            db, mlrun.common.schemas.AuthInfo(), project, schedule_name
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
    inactive_cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year="1999")
    get_db().create_schedule(
        db,
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object,
        inactive_cron_trigger,
        1,
    )

    # update schedule should fail since the schedule job was not created in the scheduler
    with pytest.raises(mlrun.errors.MLRunNotFoundError) as excinfo:
        scheduler.update_schedule(
            db, mlrun.common.schemas.AuthInfo(), project_name, schedule_name
        )
    job_id = scheduler._resolve_job_id(project_name, schedule_name)
    assert (
        f"Schedule job with id {job_id} not found in scheduler. Reload schedules is required."
        in str(excinfo.value)
    )


@pytest.mark.asyncio
# Marking the test as flaky since it depends on the scheduler to run the job in the right time.
# We were experiencing issues with concurrency_limit > 1 where some job might be unexpectedly skipped due to
# milliseconds delay. This issue is rare and if it seems to be happening more frequently, it should be addressed.
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    # The function waits 2 seconds and the schedule runs every second for 4 seconds. So:
    # For 1 concurrent job, the second and fourth jobs should be skipped resulting in 2 runs.
    # For 2 concurrent jobs, the third job should be skipped resulting in 3 runs.
    # For 3 concurrent jobs, no job should be skipped resulting in 4 runs.
    "concurrency_limit,run_amount",
    [(1, 2), (2, 3), (3, 4)],
)
@pytest.mark.parametrize(
    "schedule_kind",
    [
        mlrun.common.schemas.ScheduleKinds.job,
        mlrun.common.schemas.ScheduleKinds.local_function,
    ],
)
async def test_schedule_job_concurrency_limit(
    db: Session,
    scheduler: Scheduler,
    concurrency_limit: int,
    run_amount: int,
    schedule_kind: mlrun.common.schemas.ScheduleKinds,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    global call_counter
    call_counter = 0

    project_name = config.default_project
    create_project(db, project_name)

    scheduled_object = (
        _create_mlrun_function_and_matching_scheduled_object(
            db, project_name, handler="sleep_two_seconds"
        )
        if schedule_kind == mlrun.common.schemas.ScheduleKinds.job
        else bump_counter_and_wait
    )

    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0

    now = mlrun.utils.now_date()
    now_plus_1_seconds = now + timedelta(seconds=1)
    now_plus_5_seconds = now + timedelta(seconds=5)
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(
        second="*/1", start_date=now_plus_1_seconds, end_date=now_plus_5_seconds
    )
    schedule_name = "schedule-name"

    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project_name,
        schedule_name,
        schedule_kind,
        scheduled_object,
        cron_trigger,
        concurrency_limit=concurrency_limit,
    )

    random_sleep_time = random.randint(1, 5)
    await asyncio.sleep(random_sleep_time)
    after_sleep_timestamp = mlrun.utils.now_date()

    schedule = scheduler.get_schedule(
        db,
        project_name,
        schedule_name,
    )
    if schedule.next_run_time is None:
        # next run time may be none if the job was completed (i.e. end date was reached)
        # scrub the microseconds to reduce noise
        assert after_sleep_timestamp >= now_plus_5_seconds.replace(microsecond=0)

    else:
        # scrub the microseconds to reduce noise
        assert schedule.next_run_time >= after_sleep_timestamp.replace(microsecond=0)

    # wait so all runs will complete
    await asyncio.sleep(7 - random_sleep_time)
    if schedule_kind == mlrun.common.schemas.ScheduleKinds.job:
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
    now = mlrun.utils.now_date()
    now_plus_1_seconds = now + timedelta(seconds=1)
    now_plus_5_seconds = now + timedelta(seconds=5)
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(
        second="*/1", start_date=now_plus_1_seconds, end_date=now_plus_5_seconds
    )
    schedule_name = "schedule-name"
    project_name = config.default_project
    create_project(db, project_name)

    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name, handler="sleep_two_seconds"
    )

    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0

    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
        concurrency_limit=1,
    )

    while mlrun.utils.now_date() < now_plus_5_seconds:
        runs = get_db().list_runs(db, project=project_name)
        if len(runs) == 1:
            break

        await asyncio.sleep(0.5)
    else:
        assert False, "No runs were created"

    # invoke schedule should fail due to concurrency limit
    # the next run time should be updated to the next second after the invocation failure
    schedule_invocation_timestamp = mlrun.utils.now_date()
    await scheduler.invoke_schedule(
        db, mlrun.common.schemas.AuthInfo(), project_name, schedule_name
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


@pytest.mark.asyncio
def test_store_schedule(db: Session, scheduler: Scheduler):
    labels_1 = {
        "label1": "value1",
        "label2": "value2",
    }
    inactive_cron_trigger_1 = mlrun.common.schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "store-schedule-test"
    project_name = config.default_project

    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(
        db, project_name
    )
    runs = get_db().list_runs(db, project=project_name)
    assert len(runs) == 0
    scheduler.store_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object,
        inactive_cron_trigger_1,
        labels=labels_1,
    )

    schedule = scheduler.get_schedule(db, project_name, schedule_name)
    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        inactive_cron_trigger_1,
        None,
        labels_1,
        config.httpdb.scheduling.default_concurrency_limit,
    )

    # update labels, concurrency limit and cron trigger
    labels_2 = {
        "label3": "value3",
        "label4": "value4",
    }
    concurrency_limit = 10
    inactive_cron_trigger_2 = mlrun.common.schemas.ScheduleCronTrigger(year="2000")
    scheduler.store_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project=project_name,
        name=schedule_name,
        cron_trigger=inactive_cron_trigger_2,
        labels=labels_2,
        concurrency_limit=concurrency_limit,
    )
    schedule = scheduler.get_schedule(db, project_name, schedule_name)

    _assert_schedule(
        schedule,
        project_name,
        schedule_name,
        mlrun.common.schemas.ScheduleKinds.job,
        inactive_cron_trigger_2,
        None,
        labels_2,
        concurrency_limit,
    )


@pytest.mark.parametrize(
    "labels, scheduled_object, expected",
    [
        (
            {"key1": "value1", "key2": "value2"},
            {
                "task": {
                    "metadata": {"labels": {"key2": "new_value2", "key3": "value3"}}
                }
            },
            {"key1": "value1", "key2": "new_value2", "key3": "value3"},
        ),
        (
            {"key1": "value1"},
            {"task": {"metadata": {"labels": {}}}},
            {"key1": "value1"},
        ),
        (
            {},
            {"task": {"metadata": {"labels": {"key1": "value1"}}}},
            {"key1": "value1"},
        ),
        (
            {"key1": "value1"},
            {"task": {"metadata": {"labels": None}}},
            {"key1": "value1"},
        ),
        (
            {},
            {"task": {"metadata": {"labels": None}}},
            None,
        ),
        (
            None,
            {"task": {"metadata": {"labels": None}}},
            None,
        ),
        (
            {"key1": "value1"},
            {"task": {"metadata": {"labels": {}}}},
            {"key1": "value1"},
        ),
    ],
)
def test_merge_schedule_and_schedule_object_labels(
    scheduler, labels, scheduled_object, expected
):
    result = server.api.utils.helpers.merge_schedule_and_schedule_object_labels(
        labels,
        scheduled_object,
    )
    assert result == expected
    assert scheduled_object["task"]["metadata"]["labels"] == expected


@pytest.mark.parametrize(
    "labels, scheduled_object, db_schedule_labels, db_scheduled_object, expected",
    [
        (
            # if schedule.labels and task.labels are passed,
            # we expect to get merged values of the passed values
            {"key1": "value1", "key2": "value2"},
            {
                "task": {
                    "metadata": {"labels": {"key2": "new_value2", "key3": "value3"}}
                }
            },
            [
                mlrun.common.schemas.schedule.LabelRecord(
                    id="1", name="key1", value="db_value1"
                ),
                mlrun.common.schemas.schedule.LabelRecord(
                    id="2", name="key4", value="db_value4"
                ),
            ],
            {"task": {"metadata": {"labels": {"key1": "db_value1"}}}},
            {"key1": "value1", "key2": "new_value2", "key3": "value3"},
        ),
        (
            # if schedule.labels is passed and task.labels isn't,
            # we expect to get schedule.labels
            {"key1": "value1"},
            {"task": {"metadata": {"labels": {}}}},
            [
                mlrun.common.schemas.schedule.LabelRecord(
                    id="1", name="key1", value="db_value1"
                )
            ],
            {"task": {"metadata": {"labels": {"key1": "db_value2"}}}},
            {"key1": "value1"},
        ),
        (
            # if schedule.labels is passed and task.labels isn't,
            # we expect to get schedule.labels
            {"key1": "value1"},
            {"task": {"metadata": {"labels": None}}},
            [],
            {"task": {"metadata": {"labels": {}}}},
            {"key1": "value1"},
        ),
        (
            # if schedule.labels isn't passed and task.labels is,
            # we expect to get task.labels
            {},
            {"task": {"metadata": {"labels": {"key1": "value1"}}}},
            [],
            {"task": {"metadata": {"labels": {}}}},
            {"key1": "value1"},
        ),
        (
            # Nothing is passed, expect to get {}
            None,
            {"task": {"metadata": {"labels": None}}},
            [],
            {"task": {"metadata": {"labels": {}}}},
            {},
        ),
        (
            # if schedule.labels and task.labels are an empty dict,
            # we expect values from db to be cleaned up
            {},
            {"task": {"metadata": {"labels": {}}}},
            [
                mlrun.common.schemas.schedule.LabelRecord(
                    id="1", name="key1", value="db_value1"
                )
            ],
            {"task": {"metadata": {"labels": {}}}},
            {},
        ),
        (
            # if schedule.labels is empty, task.labels is None, and db.labels has values,
            # where db label values are different
            # we expect to get merged values from db.labels and db_task.labels
            {},
            {"task": {"metadata": {"labels": None}}},
            [
                mlrun.common.schemas.schedule.LabelRecord(
                    id="1", name="key3", value="db_value3"
                ),
                mlrun.common.schemas.schedule.LabelRecord(
                    id="2", name="key4", value="db_value4"
                ),
            ],
            {"task": {"metadata": {"labels": {"key5": "db_value5"}}}},
            {"key3": "db_value3", "key4": "db_value4", "key5": "db_value5"},
        ),
        (
            # if schedule.labels is None, task.labels is None, and db.labels has values,
            # we expect to get values from db.labels
            None,
            {"task": {"metadata": {"labels": None}}},
            [
                mlrun.common.schemas.schedule.LabelRecord(
                    id="1", name="key3", value="db_value3"
                ),
                mlrun.common.schemas.schedule.LabelRecord(
                    id="2", name="key4", value="db_value4"
                ),
            ],
            {"task": {"metadata": {"labels": {}}}},
            {"key3": "db_value3", "key4": "db_value4"},
        ),
        (
            # if schedule.labels is passed, schedule object isn't passed at all None,
            # we expect to get values from schedule.labels
            {"key1": "value1"},
            None,
            [
                mlrun.common.schemas.schedule.LabelRecord(
                    id="1", name="key3", value="db_value3"
                ),
                mlrun.common.schemas.schedule.LabelRecord(
                    id="2", name="key4", value="db_value4"
                ),
            ],
            {"task": {"metadata": {"labels": {}}}},
            {"key1": "value1"},
        ),
    ],
)
def test_merge_schedule_and_db_schedule_labels(
    scheduler,
    labels,
    scheduled_object,
    db_schedule_labels,
    db_scheduled_object,
    expected,
):
    # Create a mock of ScheduleRecord
    db_schedule = MagicMock()
    db_schedule.labels = db_schedule_labels
    db_schedule.scheduled_object = db_scheduled_object

    result_labels, result_scheduled_object = (
        server.api.utils.helpers.merge_schedule_and_db_schedule_labels(
            labels,
            scheduled_object,
            db_schedule,
        )
    )

    assert result_labels == expected
    assert result_scheduled_object["task"]["metadata"]["labels"] == expected


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

    secret_name = tests.api.conftest.K8sSecretsMock.resolve_auth_secret_name(
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
    auth_data = server.api.crud.Secrets().read_auth_secret(secret_name)
    assert expected_username == auth_data.username
    assert expected_access_key == auth_data.access_key


def _assert_schedule_secrets(
    scheduler: Scheduler,
    project: str,
    schedule_name: str,
    expected_username: str,
    expected_access_key: str,
):
    access_key_secret_key = (
        server.api.crud.Secrets().generate_client_project_secret_key(
            server.api.crud.SecretsClientType.schedules,
            schedule_name,
            scheduler._secret_access_key_subtype,
        )
    )
    username_secret_key = server.api.crud.Secrets().generate_client_project_secret_key(
        server.api.crud.SecretsClientType.schedules,
        schedule_name,
        scheduler._secret_username_subtype,
    )
    key_map_secret_key = (
        server.api.crud.Secrets().generate_client_key_map_project_secret_key(
            server.api.crud.SecretsClientType.schedules
        )
    )
    secret_value = server.api.crud.Secrets().get_project_secret(
        project,
        scheduler._secrets_provider,
        access_key_secret_key,
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )
    assert secret_value == expected_access_key
    secret_value = server.api.crud.Secrets().get_project_secret(
        project,
        scheduler._secrets_provider,
        username_secret_key,
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )
    assert secret_value == expected_username


def _assert_schedule(
    schedule: mlrun.common.schemas.ScheduleOutput,
    project: str,
    name: str,
    kind: mlrun.common.schemas.ScheduleKinds,
    cron_trigger: typing.Union[str, mlrun.common.schemas.ScheduleCronTrigger],
    next_run_time: typing.Optional[datetime] = None,
    labels: dict = None,
    concurrency_limit: int = None,
):
    assert schedule.name == name
    assert schedule.project == project
    assert schedule.kind == kind
    assert schedule.next_run_time == next_run_time
    assert schedule.cron_trigger == cron_trigger
    assert schedule.creation_time is not None
    assert DeepDiff(schedule.labels, labels, ignore_order=True) == {}
    if isinstance(schedule.scheduled_object, dict):
        # only for cases when scheduled_object is not a callable function
        assert (
            DeepDiff(
                schedule.labels,
                schedule.scheduled_object["task"]["metadata"]["labels"],
                ignore_order=True,
            )
            == {}
        )
    assert schedule.concurrency_limit == concurrency_limit


def _create_do_nothing_schedule(
    db: Session, scheduler: Scheduler, project: str, name: str
):
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year="1999")
    scheduler.create_schedule(
        db,
        mlrun.common.schemas.AuthInfo(),
        project,
        name,
        mlrun.common.schemas.ScheduleKinds.local_function,
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
    This is obviously very rare (since the times are based on mlrun.utils.now_date()) - usually the start time
    will be something like 12:08:06.100000 then the end time will be 12:08:07.10000 - meaning there will be only
     one execution on the 07 second.
    So instead of conditioning every assertion we're doing on whether the start time was a round second,
     we simply make sure it's not a round second.
    """
    now = mlrun.utils.now_date()
    if now.microsecond == 0:
        now = now + timedelta(seconds=1, milliseconds=1)
    start_date = now + timedelta(seconds=1)
    end_date = now + timedelta(seconds=1 + number_of_jobs * seconds_interval)
    return start_date, end_date
