from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Union, Callable

from apscheduler.triggers.cron import CronTrigger as APSchedulerCronTrigger
from pydantic import BaseModel


class ScheduleCronTrigger(BaseModel):
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
    timezone: Optional[str]
    jitter: Optional[str]

    def to_apscheduler_cron_trigger(self):
        return APSchedulerCronTrigger(
            self.year,
            self.month,
            self.day,
            self.week,
            self.day_of_week,
            self.hour,
            self.minute,
            self.second,
            self.start_date,
            self.end_date,
            self.timezone,
            self.jitter,
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


class Schedule(ScheduleBase):
    next_run_time: Optional[datetime]


class Schedules(BaseModel):
    schedules: List[Schedule]
