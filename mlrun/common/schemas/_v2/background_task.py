# Copyright 2026 Iguazio
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

from .._shared.background_task import BackgroundTaskState
from .._shared.object import ObjectKind


class BackgroundTaskMetadata(pydantic.BaseModel):
    name: str
    id: int | None = None
    kind: str | None = None
    project: str | None = None
    created: datetime.datetime | None = None
    updated: datetime.datetime | None = None
    timeout: int | None = None


class BackgroundTaskSpec(pydantic.BaseModel):
    pass


class BackgroundTaskStatus(pydantic.BaseModel):
    state: BackgroundTaskState
    error: str | None = None


class BackgroundTask(pydantic.BaseModel):
    kind: typing.Literal[ObjectKind.background_task] = ObjectKind.background_task
    metadata: BackgroundTaskMetadata
    spec: BackgroundTaskSpec
    status: BackgroundTaskStatus


class BackgroundTaskList(pydantic.BaseModel):
    background_tasks: list[BackgroundTask]
