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
import datetime
import typing

import pydantic.v1

import mlrun.common.types

from .object import ObjectKind


class BackgroundTaskState(mlrun.common.types.StrEnum):
    succeeded = "succeeded"
    failed = "failed"
    running = "running"

    @staticmethod
    def terminal_states():
        return [
            BackgroundTaskState.succeeded,
            BackgroundTaskState.failed,
        ]


class BackgroundTaskMetadata(pydantic.v1.BaseModel):
    name: str
    kind: typing.Optional[str]
    project: typing.Optional[str]
    created: typing.Optional[datetime.datetime]
    updated: typing.Optional[datetime.datetime]
    timeout: typing.Optional[int]


class BackgroundTaskSpec(pydantic.v1.BaseModel):
    pass


class BackgroundTaskStatus(pydantic.v1.BaseModel):
    state: BackgroundTaskState
    error: typing.Optional[str]


class BackgroundTask(pydantic.v1.BaseModel):
    kind: ObjectKind = pydantic.v1.Field(ObjectKind.background_task, const=True)
    metadata: BackgroundTaskMetadata
    spec: BackgroundTaskSpec
    status: BackgroundTaskStatus


class BackgroundTaskList(pydantic.v1.BaseModel):
    background_tasks: list[BackgroundTask]
