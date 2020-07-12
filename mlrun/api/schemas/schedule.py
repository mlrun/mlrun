from datetime import datetime
from typing import Optional, List, Dict, Union

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
    second: Optional[Union[int, str]]
    start_date: Optional[Union[datetime, str]]
    end_date: Optional[Union[datetime, str]]
    timezone: Optional[Union[datetime.tzinfo, str]]
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


# Properties to receive via API on creation
class ScheduleCreate(BaseModel):
    kind: str
    scheduled_object: Dict
    cron_trigger: ScheduleCronTrigger


class ScheduleBase(ScheduleCreate):
    creation_time: datetime
    project: str
    name: str


class ScheduleInDB(ScheduleBase):
    class Config:
        orm_mode = True


class Schedule(ScheduleBase):
    next_run_time: Optional[datetime]


class Schedules(BaseModel):
    schedules: List[Schedule]
