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

import datetime
import typing

import pydantic

import mlrun.common.types


class NotificationKind(mlrun.common.types.StrEnum):
    console = "console"
    git = "git"
    ipython = "ipython"
    slack = "slack"


class NotificationSeverity(mlrun.common.types.StrEnum):
    INFO = "info"
    DEBUG = "debug"
    VERBOSE = "verbose"
    WARNING = "warning"
    ERROR = "error"


class NotificationStatus(mlrun.common.types.StrEnum):
    PENDING = "pending"
    SENT = "sent"
    ERROR = "error"


class Notification(pydantic.BaseModel):
    kind: NotificationKind
    name: str
    message: str
    severity: NotificationSeverity
    when: typing.List[str]
    condition: str
    params: typing.Dict[str, typing.Any] = None
    status: NotificationStatus = None
    sent_time: typing.Union[str, datetime.datetime] = None


class SetNotificationRequest(pydantic.BaseModel):
    notifications: typing.List[Notification] = None
