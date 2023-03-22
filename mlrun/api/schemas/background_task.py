# Copyright 2018 Iguazio
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

import pydantic

import mlrun.api.utils.helpers

from .object import ObjectKind


class BackgroundTaskState(mlrun.api.utils.helpers.StrEnum):
    succeeded = "succeeded"
    failed = "failed"
    running = "running"

    @staticmethod
    def terminal_states():
        return [
            BackgroundTaskState.succeeded,
            BackgroundTaskState.failed,
        ]


class BackgroundTaskMetadata(pydantic.BaseModel):
    name: str
    project: typing.Optional[str]
    created: typing.Optional[datetime.datetime]
    updated: typing.Optional[datetime.datetime]
    timeout: typing.Optional[int]


class BackgroundTaskSpec(pydantic.BaseModel):
    pass


class BackgroundTaskStatus(pydantic.BaseModel):
    state: BackgroundTaskState


class BackgroundTask(pydantic.BaseModel):
    kind: ObjectKind = pydantic.Field(ObjectKind.background_task, const=True)
    metadata: BackgroundTaskMetadata
    spec: BackgroundTaskSpec
    status: BackgroundTaskStatus
