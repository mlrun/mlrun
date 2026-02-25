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

import typing

import pydantic.v1

from mlrun.common.schemas.notification import Notification
from mlrun.common.schemas.schedule import ScheduleCronTrigger
from mlrun.common.types import StrEnum


class WorkflowSpec(pydantic.v1.BaseModel):
    name: str
    engine: str | None = None
    code: str | None = None
    path: str | None = None
    args: dict | None = None
    handler: str | None = None
    ttl: int | None = None
    args_schema: list | None = None
    schedule: typing.Union[str, ScheduleCronTrigger] = None
    run_local: bool | None = None
    image: str | None = None
    workflow_runner_node_selector: dict[str, str] | None = None
    auth_token_name: str | None = None


class WorkflowRequest(pydantic.v1.BaseModel):
    spec: WorkflowSpec | None = None
    arguments: dict | None = None
    artifact_path: str | None = None
    source: str | None = None
    run_name: str | None = None
    namespace: str | None = None
    notifications: list[Notification] | None = None


class RerunWorkflowRequest(pydantic.v1.BaseModel):
    run_name: str | None = None
    run_id: str | None = None
    notifications: list[Notification] | None = None
    workflow_runner_node_selector: dict[str, str] | None = None
    original_workflow_runner_uid: str | None = None
    original_workflow_name: str | None = None
    rerun_index: int | None = None


class WorkflowResponse(pydantic.v1.BaseModel):
    project: str = None
    name: str = None
    status: str = None
    run_id: str | None = None
    schedule: typing.Union[str, ScheduleCronTrigger] = None


class GetWorkflowResponse(pydantic.v1.BaseModel):
    workflow_id: str = None


class EngineType(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"
    KFP = "kfp"
    REMOTE_KFP = "remote:kfp"
