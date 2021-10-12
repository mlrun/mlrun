import datetime
import enum
import typing

import pydantic

from .object import ObjectKind, ObjectStatus


class ProjectsFormat(str, enum.Enum):
    full = "full"
    name_only = "name_only"
    # internal - allowed only in follower mode, only for the leader for upgrade purposes
    leader = "leader"


class ProjectMetadata(pydantic.BaseModel):
    name: str
    created: typing.Optional[datetime.datetime] = None
    labels: typing.Optional[dict]
    annotations: typing.Optional[dict]

    class Config:
        extra = pydantic.Extra.allow


class ProjectDesiredState(str, enum.Enum):
    online = "online"
    offline = "offline"
    archived = "archived"


class ProjectState(str, enum.Enum):
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


class ProjectSpec(pydantic.BaseModel):
    description: typing.Optional[str] = None
    owner: typing.Optional[str] = None
    goals: typing.Optional[str] = None
    params: typing.Optional[dict] = None
    functions: typing.Optional[list] = None
    workflows: typing.Optional[list] = None
    artifacts: typing.Optional[list] = None
    artifact_path: typing.Optional[str] = None
    conda: typing.Optional[str] = None
    source: typing.Optional[str] = None
    subpath: typing.Optional[str] = None
    origin_url: typing.Optional[str] = None
    desired_state: typing.Optional[ProjectDesiredState] = ProjectDesiredState.online

    class Config:
        extra = pydantic.Extra.allow


class Project(pydantic.BaseModel):
    kind: ObjectKind = pydantic.Field(ObjectKind.project, const=True)
    metadata: ProjectMetadata
    spec: ProjectSpec = ProjectSpec()
    status: ObjectStatus = ObjectStatus()


class ProjectOwner(pydantic.BaseModel):
    username: str
    session: str


class ProjectSummary(pydantic.BaseModel):
    name: str
    files_count: int
    feature_sets_count: int
    models_count: int
    runs_failed_recent_count: int
    runs_running_count: int
    schedules_count: int
    pipelines_running_count: int


class IguazioProject(pydantic.BaseModel):
    data: dict


class ProjectsOutput(pydantic.BaseModel):
    # The format query param controls the project type used:
    # full - Project
    # name_only - str
    # summary - ProjectSummary
    # leader - currently only IguazioProject supported
    # The way pydantic handles typing.Union is that it takes the object and tries to coerce it to be the types of the
    # union by the definition order. Therefore we can't currently add generic dict for all leader formats, but we need
    # to add a specific classes for them. it's frustrating but couldn't find other workaround, see:
    # https://github.com/samuelcolvin/pydantic/issues/1423, https://github.com/samuelcolvin/pydantic/issues/619
    projects: typing.List[typing.Union[Project, str, ProjectSummary, IguazioProject]]


class ProjectSummariesOutput(pydantic.BaseModel):
    project_summaries: typing.List[ProjectSummary]
