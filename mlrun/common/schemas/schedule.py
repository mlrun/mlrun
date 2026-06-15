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

from datetime import datetime
from typing import Any, Literal, Union

from pydantic.v1 import BaseModel

import mlrun.common.types
from mlrun.common.schemas.auth import Credentials
from mlrun.common.schemas.object import LabelRecord


class ScheduleCronTrigger(BaseModel):
    """
    See this link for help
    https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html#module-apscheduler.triggers.cron
    """

    year: Union[int, str] | None
    month: Union[int, str] | None
    day: Union[int, str] | None
    week: Union[int, str] | None
    day_of_week: Union[int, str] | None
    hour: Union[int, str] | None
    minute: Union[int, str] | None
    second: Union[int, str] | None
    start_date: Union[datetime, str] = None
    end_date: Union[datetime, str] = None

    # APScheduler also supports datetime.tzinfo type, but Pydantic doesn't - so we don't
    timezone: str | None
    jitter: int | None

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
    scheduled_object: Any | None
    cron_trigger: Union[str, ScheduleCronTrigger] | None
    desired_state: str | None
    labels: dict | None = None
    concurrency_limit: int | None
    credentials: Credentials = Credentials()


# Properties to receive via API on creation
class ScheduleInput(BaseModel):
    name: str
    kind: ScheduleKinds
    scheduled_object: Any
    cron_trigger: Union[str, ScheduleCronTrigger]
    desired_state: str | None
    labels: dict | None = {}
    concurrency_limit: int | None
    credentials: Credentials = Credentials()


# the schedule object returned from the db layer
class ScheduleRecord(ScheduleInput):
    creation_time: datetime
    project: str
    last_run_uri: str | None
    state: str | None
    labels: list[LabelRecord] | None
    next_run_time: datetime | None

    class Config:
        orm_mode = True


# Additional properties to return via API
class ScheduleOutput(ScheduleRecord):
    next_run_time: datetime | None
    last_run: dict | None = {}
    labels: dict | None = {}
    credentials: Credentials = Credentials()


class SchedulesOutput(BaseModel):
    schedules: list[ScheduleOutput]


class ScheduleIdentifier(BaseModel):
    kind: Literal["schedule"] = "schedule"
    name: str
