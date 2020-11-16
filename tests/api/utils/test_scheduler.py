import asyncio
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Generator
from dateutil.tz import tzlocal

import pytest
from deepdiff import DeepDiff
from sqlalchemy.orm import Session

import mlrun
from mlrun.api import schemas
from mlrun.api.utils.scheduler import Scheduler
from mlrun.api.utils.singletons.db import get_db
from mlrun.config import config
from mlrun.runtimes.base import RunStates
from mlrun.utils import logger


@pytest.fixture()
async def scheduler(db: Session) -> Generator:
    logger.info("Creating scheduler")
    config.httpdb.scheduling.min_allowed_interval = "0"
    scheduler = Scheduler()
    await scheduler.start(db)
    yield scheduler
    logger.info("Stopping scheduler")
    await scheduler.stop()


call_counter: int = 0


async def bump_counter():
    global call_counter
    call_counter += 1


async def do_nothing():
    pass


@pytest.mark.asyncio
async def test_create_schedule(db: Session, scheduler: Scheduler):
    global call_counter
    call_counter = 0
    now = datetime.now()
    expected_call_counter = 5
    now_plus_1_seconds = now + timedelta(seconds=1)
    now_plus_5_seconds = now + timedelta(seconds=1 + expected_call_counter)
    # this way we're leaving ourselves one second to create the schedule preventing transient test failure
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1", start_date=now_plus_1_seconds, end_date=now_plus_5_seconds
    )
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        bump_counter,
        cron_trigger,
    )
    await asyncio.sleep(1 + expected_call_counter)
    assert call_counter == expected_call_counter


@pytest.mark.asyncio
async def test_invoke_schedule(db: Session, scheduler: Scheduler):
    cron_trigger = schemas.ScheduleCronTrigger(year=1999)
    schedule_name = "schedule-name"
    project = config.default_project
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(db, project)
    runs = get_db().list_runs(db, project=project)
    assert len(runs) == 0
    scheduler.create_schedule(
        db,
        project,
        schedule_name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )
    runs = get_db().list_runs(db, project=project)
    assert len(runs) == 0
    response_1 = await scheduler.invoke_schedule(db, project, schedule_name)
    runs = get_db().list_runs(db, project=project)
    assert len(runs) == 1
    response_2 = await scheduler.invoke_schedule(db, project, schedule_name)
    runs = get_db().list_runs(db, project=project)
    assert len(runs) == 2
    for run in runs:
        assert run["status"]["state"] == RunStates.completed
    response_uids = [
        response["data"]["metadata"]["uid"] for response in [response_1, response_2]
    ]
    db_uids = [run["metadata"]["uid"] for run in runs]
    assert DeepDiff(response_uids, db_uids, ignore_order=True,) == {}

    schedule = scheduler.get_schedule(db, project, schedule_name, include_last_run=True)
    assert schedule.last_run is not None
    assert schedule.last_run["metadata"]["uid"] == response_uids[-1]
    assert schedule.last_run["metadata"]["project"] == project


@pytest.mark.asyncio
async def test_create_schedule_mlrun_function(db: Session, scheduler: Scheduler):
    now = datetime.now()
    now_plus_1_second = now + timedelta(seconds=1)
    now_plus_2_second = now + timedelta(seconds=2)
    # this way we're leaving ourselves one second to create the schedule preventing transient test failure
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1", start_date=now_plus_1_second, end_date=now_plus_2_second
    )
    schedule_name = "schedule-name"
    project = config.default_project
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(db, project)
    runs = get_db().list_runs(db, project=project)
    assert len(runs) == 0
    scheduler.create_schedule(
        db,
        project,
        schedule_name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        cron_trigger,
    )
    await asyncio.sleep(2)
    runs = get_db().list_runs(db, project=project)
    assert len(runs) == 1
    assert runs[0]["status"]["state"] == RunStates.completed

    expected_last_run_uri = f"{project}@{runs[0]['metadata']['uid']}#0"

    schedule = get_db().get_schedule(db, project, schedule_name)
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
            "project",
            f"schedule-name-{index}",
            schemas.ScheduleKinds.local_function,
            do_nothing,
            cron_trigger,
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
                "project",
                "schedule-name",
                schemas.ScheduleKinds.local_function,
                do_nothing,
                cron_trigger,
            )
        assert "Cron trigger too frequent. no more then one job" in str(excinfo.value)


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
            project,
            name,
            schemas.ScheduleKinds.local_function,
            do_nothing,
            cron_trigger,
        )
        if should_find:
            expected_schedule_names.append(name)

    schedules = scheduler.list_schedules(db, project, "mlrun")
    assert len(schedules.schedules) == len(expected_schedule_names)
    for schedule in schedules.schedules:
        assert schedule.name in expected_schedule_names
        expected_schedule_names.remove(schedule.name)


@pytest.mark.asyncio
async def test_delete_schedule(db: Session, scheduler: Scheduler):
    cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
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


@pytest.mark.asyncio
async def test_rescheduling(db: Session, scheduler: Scheduler):
    global call_counter
    call_counter = 0
    now = datetime.now()
    now_plus_2_seconds = now + timedelta(seconds=2)
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1", start_date=now, end_date=now_plus_2_seconds
    )
    schedule_name = "schedule-name"
    project = config.default_project
    scheduler.create_schedule(
        db,
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        bump_counter,
        cron_trigger,
    )

    # wait so one run will complete
    await asyncio.sleep(1)

    # stop the scheduler and assert indeed only one call happened
    await scheduler.stop()
    assert call_counter == 1

    # start the scheduler and and assert another run
    await scheduler.start(db)
    await asyncio.sleep(1)
    assert call_counter == 2


@pytest.mark.asyncio
async def test_update_schedule(db: Session, scheduler: Scheduler):
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
    project = config.default_project
    scheduled_object = _create_mlrun_function_and_matching_scheduled_object(db, project)
    runs = get_db().list_runs(db, project=project)
    assert len(runs) == 0
    scheduler.create_schedule(
        db,
        project,
        schedule_name,
        schemas.ScheduleKinds.job,
        scheduled_object,
        inactive_cron_trigger,
        labels=labels_1,
    )

    schedule = scheduler.get_schedule(db, project, schedule_name)

    _assert_schedule(
        schedule,
        project,
        schedule_name,
        schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        labels_1,
    )

    # update labels
    scheduler.update_schedule(
        db, project, schedule_name, labels=labels_2,
    )
    schedule = scheduler.get_schedule(db, project, schedule_name)

    _assert_schedule(
        schedule,
        project,
        schedule_name,
        schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        labels_2,
    )

    # update nothing
    scheduler.update_schedule(
        db, project, schedule_name,
    )
    schedule = scheduler.get_schedule(db, project, schedule_name)

    _assert_schedule(
        schedule,
        project,
        schedule_name,
        schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        labels_2,
    )

    # update labels to empty dict
    scheduler.update_schedule(
        db, project, schedule_name, labels={},
    )
    schedule = scheduler.get_schedule(db, project, schedule_name)

    _assert_schedule(
        schedule,
        project,
        schedule_name,
        schemas.ScheduleKinds.job,
        inactive_cron_trigger,
        None,
        {},
    )

    # update it so it runs
    now = datetime.now()
    now_plus_1_second = now + timedelta(seconds=1)
    now_plus_2_second = now + timedelta(seconds=2)
    # this way we're leaving ourselves one second to create the schedule preventing transient test failure
    cron_trigger = schemas.ScheduleCronTrigger(
        second="*/1", start_date=now_plus_1_second, end_date=now_plus_2_second,
    )
    scheduler.update_schedule(
        db, project, schedule_name, cron_trigger=cron_trigger,
    )
    schedule = scheduler.get_schedule(db, project, schedule_name)

    next_run_time = datetime(
        year=now_plus_2_second.year,
        month=now_plus_2_second.month,
        day=now_plus_2_second.day,
        hour=now_plus_2_second.hour,
        minute=now_plus_2_second.minute,
        second=now_plus_2_second.second,
        tzinfo=tzlocal(),
    )

    _assert_schedule(
        schedule,
        project,
        schedule_name,
        schemas.ScheduleKinds.job,
        cron_trigger,
        next_run_time,
        {},
    )

    await asyncio.sleep(2)
    runs = get_db().list_runs(db, project=project)
    assert len(runs) == 1
    assert runs[0]["status"]["state"] == RunStates.completed


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
    assert len(schedule.labels) == len(labels)
    for label in schedule.labels:
        assert labels[label.name] == label.value


def _create_mlrun_function_and_matching_scheduled_object(db: Session, project: str):
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
                "handler": "do_nothing",
            },
            "metadata": {"name": "my-task", "project": f"{project}"},
        }
    }
    return scheduled_object
