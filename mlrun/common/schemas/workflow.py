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
import typing

import pydantic

from mlrun.common.schemas.notification import Notification
from mlrun.common.schemas.schedule import ScheduleCronTrigger
from mlrun.common.types import StrEnum


class WorkflowSpec(pydantic.BaseModel):
    name: str
    engine: typing.Optional[str] = None
    code: typing.Optional[str] = None
    path: typing.Optional[str] = None
    args: typing.Optional[dict] = None
    handler: typing.Optional[str] = None
    ttl: typing.Optional[int] = None
    args_schema: typing.Optional[list] = None
    schedule: typing.Union[str, ScheduleCronTrigger] = None
    run_local: typing.Optional[bool] = None
    image: typing.Optional[str] = None
    workflow_runner_node_selector: typing.Optional[dict[str, str]] = None


class WorkflowRequest(pydantic.BaseModel):
    spec: typing.Optional[WorkflowSpec] = None
    arguments: typing.Optional[dict] = None
    artifact_path: typing.Optional[str] = None
    source: typing.Optional[str] = None
    run_name: typing.Optional[str] = None
    namespace: typing.Optional[str] = None
    notifications: typing.Optional[list[Notification]] = None


class WorkflowResponse(pydantic.BaseModel):
    project: str = None
    name: str = None
    status: str = None
    run_id: typing.Optional[str] = None
    schedule: typing.Union[str, ScheduleCronTrigger] = None


class GetWorkflowResponse(pydantic.BaseModel):
    workflow_id: str = None


class EngineType(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"
    KFP = "kfp"
