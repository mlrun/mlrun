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
import uuid

import pydantic.v1

import mlrun.common.types

from .common import ImageBuilder
from .object import ObjectKind, ObjectStatus


class ProjectMetadata(pydantic.v1.BaseModel):
    name: str
    created: datetime.datetime | None = None
    labels: dict | None = {}
    annotations: dict | None = {}

    class Config:
        extra = pydantic.v1.Extra.allow


class ProjectDesiredState(mlrun.common.types.StrEnum):
    online = "online"
    offline = "offline"
    archived = "archived"


class ProjectState(mlrun.common.types.StrEnum):
    unknown = "unknown"
    creating = "creating"
    deleting = "deleting"
    online = "online"
    offline = "offline"
    archived = "archived"

    @staticmethod
    def terminal_states():
        return [
            ProjectState.online,
            ProjectState.offline,
            ProjectState.archived,
        ]


class ProjectStatus(ObjectStatus):
    state: ProjectState | None
    op_id: uuid.UUID | None = None
    phase: int | None = None
    updated_at: datetime.datetime | None = None


class ProjectSpec(pydantic.v1.BaseModel):
    description: str | None = None
    owner: str | None = None
    goals: str | None = None
    params: dict | None = {}
    functions: list | None = []
    workflows: list | None = []
    artifacts: list | None = []
    artifact_path: str | None = None
    conda: str | None = None
    source: str | None = None
    subpath: str | None = None
    origin_url: str | None = None
    desired_state: ProjectDesiredState | None = ProjectDesiredState.online
    custom_packagers: list[tuple[str, bool]] | None = None
    default_image: str | None = None
    build: ImageBuilder | None = None
    default_function_node_selector: dict | None = {}

    class Config:
        extra = pydantic.v1.Extra.allow


class ProjectSpecOut(pydantic.v1.BaseModel):
    description: str | None = None
    owner: str | None = None
    goals: str | None = None
    params: dict | None = {}
    functions: list | None = []
    workflows: list | None = []
    artifacts: list | None = []
    artifact_path: str | None = None
    conda: str | None = None
    source: str | None = None
    subpath: str | None = None
    origin_url: str | None = None
    desired_state: ProjectDesiredState | None = ProjectDesiredState.online
    custom_packagers: list[tuple[str, bool]] | None = None
    default_image: str | None = None
    build: typing.Any = None
    default_function_node_selector: dict | None = {}

    class Config:
        extra = pydantic.v1.Extra.allow


class Project(pydantic.v1.BaseModel):
    kind: ObjectKind = pydantic.v1.Field(ObjectKind.project, const=True)
    metadata: ProjectMetadata
    spec: ProjectSpec = ProjectSpec()
    status: ProjectStatus = ProjectStatus()


# The reason we have a different schema for the response model is that we don't want to validate project.spec.build in
# the response as the validation was added late and there may be corrupted values in the DB.
class ProjectOut(pydantic.v1.BaseModel):
    kind: ObjectKind = pydantic.v1.Field(ObjectKind.project, const=True)
    metadata: ProjectMetadata
    spec: ProjectSpecOut = ProjectSpecOut()
    status: ProjectStatus = ProjectStatus()


class ProjectOwner(pydantic.v1.BaseModel):
    username: str
    access_key: str


class ProjectSummary(pydantic.v1.BaseModel):
    name: str
    files_count: int = 0
    feature_sets_count: int = 0
    models_count: int = 0
    runs_completed_recent_count: int = 0
    runs_failed_recent_count: int = 0
    runs_running_count: int = 0
    distinct_schedules_count: int = 0
    distinct_scheduled_jobs_pending_count: int = 0
    distinct_scheduled_pipelines_pending_count: int = 0
    pipelines_completed_recent_count: int = 0
    pipelines_failed_recent_count: int = 0
    pipelines_running_count: int = 0
    updated: datetime.datetime | None = None
    endpoint_alerts_count: int = 0
    job_alerts_count: int = 0
    application_alerts_count: int = 0
    infra_alerts_count: int = 0
    datasets_count: int = 0
    documents_count: int = 0
    llm_prompts_count: int = 0
    running_model_monitoring_functions: int = 0
    failed_model_monitoring_functions: int = 0
    real_time_model_endpoint_count: int = 0
    batch_model_endpoint_count: int = 0


class IguazioProject(pydantic.v1.BaseModel):
    data: dict


# The format query param controls the project type used:
# full - ProjectOut
# name_only - str
# summary - ProjectSummary
# leader - currently only IguazioProject supported
# The way pydantic handles typing.Union is that it takes the object and tries to coerce it to be the types of the
# union by the definition order. Therefore, we can't currently add generic dict for all leader formats, but we need
# to add a specific classes for them. it's frustrating but couldn't find other workaround, see:
# https://github.com/samuelcolvin/pydantic/issues/1423, https://github.com/samuelcolvin/pydantic/issues/619
ProjectOutput = typing.TypeVar(
    "ProjectOutput",
    ProjectOut,
    str,
    ProjectSummary,
    IguazioProject,
    tuple[str, datetime.datetime],
)


class ProjectsOutput(pydantic.v1.BaseModel):
    projects: list[ProjectOutput]


class ProjectSummariesOutput(pydantic.v1.BaseModel):
    project_summaries: list[ProjectSummary]
