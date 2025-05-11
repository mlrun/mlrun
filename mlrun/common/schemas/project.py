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
from deprecated import deprecated

import mlrun.common.types

from .common import ImageBuilder
from .object import ObjectKind, ObjectStatus


@deprecated(
    version="1.7.0",
    reason="mlrun.common.schemas.ProjectsFormat is deprecated and will be removed in 1.10.0. "
    "Use mlrun.common.formatters.ProjectFormat instead.",
    category=FutureWarning,
)
class ProjectsFormat(mlrun.common.types.StrEnum):
    full = "full"
    name_only = "name_only"
    # minimal format removes large fields from the response (e.g. functions, workflows, artifacts)
    # and is used for faster response times (in the UI)
    minimal = "minimal"
    # internal - allowed only in follower mode, only for the leader for upgrade purposes
    leader = "leader"


class ProjectMetadata(pydantic.v1.BaseModel):
    name: str
    created: typing.Optional[datetime.datetime] = None
    labels: typing.Optional[dict] = {}
    annotations: typing.Optional[dict] = {}

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
    state: typing.Optional[ProjectState]


class ProjectSpec(pydantic.v1.BaseModel):
    description: typing.Optional[str] = None
    owner: typing.Optional[str] = None
    goals: typing.Optional[str] = None
    params: typing.Optional[dict] = {}
    functions: typing.Optional[list] = []
    workflows: typing.Optional[list] = []
    artifacts: typing.Optional[list] = []
    artifact_path: typing.Optional[str] = None
    conda: typing.Optional[str] = None
    source: typing.Optional[str] = None
    subpath: typing.Optional[str] = None
    origin_url: typing.Optional[str] = None
    desired_state: typing.Optional[ProjectDesiredState] = ProjectDesiredState.online
    custom_packagers: typing.Optional[list[tuple[str, bool]]] = None
    default_image: typing.Optional[str] = None
    build: typing.Optional[ImageBuilder] = None
    default_function_node_selector: typing.Optional[dict] = {}

    class Config:
        extra = pydantic.v1.Extra.allow


class ProjectSpecOut(pydantic.v1.BaseModel):
    description: typing.Optional[str] = None
    owner: typing.Optional[str] = None
    goals: typing.Optional[str] = None
    params: typing.Optional[dict] = {}
    functions: typing.Optional[list] = []
    workflows: typing.Optional[list] = []
    artifacts: typing.Optional[list] = []
    artifact_path: typing.Optional[str] = None
    conda: typing.Optional[str] = None
    source: typing.Optional[str] = None
    subpath: typing.Optional[str] = None
    origin_url: typing.Optional[str] = None
    desired_state: typing.Optional[ProjectDesiredState] = ProjectDesiredState.online
    custom_packagers: typing.Optional[list[tuple[str, bool]]] = None
    default_image: typing.Optional[str] = None
    build: typing.Any = None
    default_function_node_selector: typing.Optional[dict] = {}

    class Config:
        extra = pydantic.v1.Extra.allow


class Project(pydantic.v1.BaseModel):
    kind: ObjectKind = pydantic.v1.Field(ObjectKind.project, const=True)
    metadata: ProjectMetadata
    spec: ProjectSpec = ProjectSpec()
    status: ObjectStatus = ObjectStatus()


# The reason we have a different schema for the response model is that we don't want to validate project.spec.build in
# the response as the validation was added late and there may be corrupted values in the DB.
class ProjectOut(pydantic.v1.BaseModel):
    kind: ObjectKind = pydantic.v1.Field(ObjectKind.project, const=True)
    metadata: ProjectMetadata
    spec: ProjectSpecOut = ProjectSpecOut()
    status: ObjectStatus = ObjectStatus()


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
    pipelines_completed_recent_count: typing.Optional[int] = None
    pipelines_failed_recent_count: typing.Optional[int] = None
    pipelines_running_count: typing.Optional[int] = None
    updated: typing.Optional[datetime.datetime] = None
    endpoint_alerts_count: int = 0
    job_alerts_count: int = 0
    other_alerts_count: int = 0


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
