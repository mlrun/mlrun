from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Union, Callable

from pydantic import BaseModel


class ScheduleCronTrigger(BaseModel):
    """
    See this link for help
    https://apscheduler.readthedocs.io/en/v3.6.3/modules/triggers/cron.html#module-apscheduler.triggers.cron
    """

    year: Optional[Union[int, str]]
    month: Optional[Union[int, str]]
    day: Optional[Union[int, str]]
    week: Optional[Union[int, str]]
    day_of_week: Optional[Union[int, str]]
    hour: Optional[Union[int, str]]
    minute: Optional[Union[int, str]]
    second: Optional[Union[int, str]]
    start_date: Optional[Union[datetime, str]]
    end_date: Optional[Union[datetime, str]]

    # APScheduler also supports datetime.tzinfo type, but Pydantic doesn't - so we don't
    timezone: Optional[str]
    jitter: Optional[int]

    @classmethod
    def from_crontab(cls, expr, timezone=None):
        """
        Create a :class:`~ScheduleCronTrigger` from a standard crontab expression.

        See https://en.wikipedia.org/wiki/Cron for more information on the format accepted here.

        :param expr: minute, hour, day of month, month, day of week
        :param datetime.tzinfo|str timezone: time zone to use for the date/time calculations (
            defaults to scheduler timezone)
        :return: a :class:`~ScheduleCronTrigger` instance

        """
        values = expr.split()
        if len(values) != 5:
            raise ValueError(
                'Wrong number of fields; got {}, expected 5'.format(len(values))
            )

        return cls(
            minute=values[0],
            hour=values[1],
            day=values[2],
            month=values[3],
            day_of_week=values[4],
            timezone=timezone,
        )


class ScheduledObjectKinds(str, Enum):
    job = "job"
    pipeline = "pipeline"

    # this is mainly for testing purposes
    local_function = "local_function"


# Properties to receive via API on creation
class ScheduleCreate(BaseModel):
    name: str
    kind: ScheduledObjectKinds
    scheduled_object: Union[Dict, Callable]
    cron_trigger: ScheduleCronTrigger


class ScheduleBase(ScheduleCreate):
    creation_time: datetime
    project: str


class ScheduleInDB(ScheduleBase):
    class Config:
        orm_mode = True


# Additional properties to return via API
class Schedule(ScheduleBase):
    next_run_time: Optional[datetime]


class Schedules(BaseModel):
    schedules: List[Schedule]
