import asyncio
from datetime import datetime, timedelta, timezone
from typing import Generator

import pytest
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.utils.scheduler import Scheduler
from mlrun.config import config
from mlrun.utils import logger


@pytest.fixture()
async def scheduler(db: Session) -> Generator:
    logger.info("Creating scheduler")
    config.httpdb.minimum_interval_between_scheduled_jobs = '0'
    scheduler = Scheduler()
    await scheduler.start(db)
    yield scheduler
    logger.info("Stopping scheduler")
    await scheduler.stop()


call_counter: int = 0


def bump_counter():
    global call_counter
    call_counter += 1


def do_nothing():
    pass


@pytest.mark.asyncio
async def test_create_schedule(db: Session, scheduler: Scheduler):
    global call_counter
    call_counter = 0
    now = datetime.now()
    expected_call_counter = 5
    now_plus_5_seconds = now + timedelta(seconds=expected_call_counter)
    cron_trigger = schemas.ScheduleCronTrigger(
        second='*/1', start_time=now, end_time=now_plus_5_seconds
    )
    schedule_name = 'schedule-name'
    project = config.default_project
    scheduler.create_schedule(
        db,
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        bump_counter,
        cron_trigger,
    )
    await asyncio.sleep(expected_call_counter)
    assert call_counter == expected_call_counter


@pytest.mark.asyncio
async def test_create_schedule_success_cron_trigger_validation(
    db: Session, scheduler: Scheduler
):
    scheduler._minimum_interval_between_jobs_minutes = 10
    cases = [
        {'second': '1', 'minute': '19'},
        {'second': '30', 'minute': '9,19'},
        {'minute': '*/10'},
        {'minute': '20-40/10'},
        {'hour': '1'},
        {'year': '1999'},
        {'year': '2050'},
    ]
    for index, case in enumerate(cases):
        cron_trigger = schemas.ScheduleCronTrigger(**case)
        scheduler.create_schedule(
            db,
            'project',
            f'schedule-name-{index}',
            schemas.ScheduleKinds.local_function,
            do_nothing,
            cron_trigger,
        )


@pytest.mark.asyncio
async def test_create_schedule_failure_too_frequent_cron_trigger(
    db: Session, scheduler: Scheduler
):
    scheduler._minimum_interval_between_jobs_minutes = 10
    cases = [
        {'second': '*'},
        {'second': '1,2'},
        {'second': '*/30'},
        {'second': '30-35'},
        {'second': '30-40/5'},
        {'minute': '*'},
        {'minute': '*'},
        {'minute': '*/5'},
        {'minute': '43-59'},
        {'minute': '30-50/6'},
        {'minute': '1,3,5'},
    ]
    for case in cases:
        cron_trigger = schemas.ScheduleCronTrigger(**case)
        with pytest.raises(ValueError) as excinfo:
            scheduler.create_schedule(
                db,
                'project',
                'schedule-name',
                schemas.ScheduleKinds.local_function,
                do_nothing,
                cron_trigger,
            )
        assert 'Cron trigger too frequent. no more then one job' in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_schedule(db: Session, scheduler: Scheduler):
    cron_trigger = schemas.ScheduleCronTrigger(year='1999')
    schedule_name = 'schedule-name'
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

    # no next run time cause we put year=1999
    assert_schedule(
        schedule,
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        cron_trigger,
        None,
    )

    year = 2050
    cron_trigger_2 = schemas.ScheduleCronTrigger(year=year, timezone='utc')
    schedule_name_2 = 'schedule-name-2'
    scheduler.create_schedule(
        db,
        project,
        schedule_name_2,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger_2,
    )
    schedule_2 = scheduler.get_schedule(db, project, schedule_name_2)
    year_datetime = datetime(year=year, month=1, day=1, tzinfo=timezone.utc)
    assert_schedule(
        schedule_2,
        project,
        schedule_name_2,
        schemas.ScheduleKinds.local_function,
        cron_trigger_2,
        year_datetime,
    )

    schedules = scheduler.list_schedules(db)
    assert len(schedules.schedules) == 2
    assert_schedule(
        schedules.schedules[0],
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        cron_trigger,
        None,
    )
    assert_schedule(
        schedules.schedules[1],
        project,
        schedule_name_2,
        schemas.ScheduleKinds.local_function,
        cron_trigger_2,
        year_datetime,
    )


@pytest.mark.asyncio
async def test_delete_schedule(db: Session, scheduler: Scheduler):
    cron_trigger = schemas.ScheduleCronTrigger(year='1999')
    schedule_name = 'schedule-name'
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
        second='*/1', start_time=now, end_time=now_plus_2_seconds
    )
    schedule_name = 'schedule-name'
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


def assert_schedule(
    schedule: schemas.ScheduleOutput, project, name, kind, cron_trigger, next_run_time
):
    assert schedule.name == name
    assert schedule.project == project
    assert schedule.kind == kind
    assert schedule.next_run_time == next_run_time
    assert schedule.cron_trigger == cron_trigger
    assert schedule.creation_time is not None
