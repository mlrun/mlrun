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
from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic.v1 import BaseModel

import mlrun.common.types
from mlrun.common.schemas.auth import Credentials
from mlrun.common.schemas.object import LabelRecord


class ScheduleCronTrigger(BaseModel):
    """
    See this link for help
    https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html#module-apscheduler.triggers.cron
    """

    year: Optional[Union[int, str]]
    month: Optional[Union[int, str]]
    day: Optional[Union[int, str]]
    week: Optional[Union[int, str]]
    day_of_week: Optional[Union[int, str]]
    hour: Optional[Union[int, str]]
    minute: Optional[Union[int, str]]
    second: Optional[Union[int, str]]
    start_date: Union[datetime, str] = None
    end_date: Union[datetime, str] = None

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
                f"Wrong number of fields in crontab expression; got {len(values)}, expected 5"
            )

        return cls(
            minute=values[0],
            hour=values[1],
            day=values[2],
            month=values[3],
            day_of_week=values[4],
            timezone=timezone,
        )

    def to_crontab(self) -> str:
        """
        Convert the trigger to a crontab expression.
        """
        return f"{self.minute} {self.hour} {self.day} {self.month} {self.day_of_week}"


class ScheduleKinds(mlrun.common.types.StrEnum):
    job = "job"
    pipeline = "pipeline"

    # this is mainly for testing purposes
    local_function = "local_function"

    @staticmethod
    def local_kinds():
        return [
            ScheduleKinds.local_function,
        ]


class ScheduleUpdate(BaseModel):
    scheduled_object: Optional[Any]
    cron_trigger: Optional[Union[str, ScheduleCronTrigger]]
    desired_state: Optional[str]
    labels: Optional[dict] = None
    concurrency_limit: Optional[int]
    credentials: Credentials = Credentials()


# Properties to receive via API on creation
class ScheduleInput(BaseModel):
    name: str
    kind: ScheduleKinds
    scheduled_object: Any
    cron_trigger: Union[str, ScheduleCronTrigger]
    desired_state: Optional[str]
    labels: Optional[dict] = {}
    concurrency_limit: Optional[int]
    credentials: Credentials = Credentials()


# the schedule object returned from the db layer
class ScheduleRecord(ScheduleInput):
    creation_time: datetime
    project: str
    last_run_uri: Optional[str]
    state: Optional[str]
    labels: Optional[list[LabelRecord]]
    next_run_time: Optional[datetime]

    class Config:
        orm_mode = True


# Additional properties to return via API
class ScheduleOutput(ScheduleRecord):
    next_run_time: Optional[datetime]
    last_run: Optional[dict] = {}
    labels: Optional[dict] = {}
    credentials: Credentials = Credentials()


class SchedulesOutput(BaseModel):
    schedules: list[ScheduleOutput]


class ScheduleIdentifier(BaseModel):
    kind: Literal["schedule"] = "schedule"
    name: str
