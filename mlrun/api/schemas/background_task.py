import datetime
import enum
import typing

import pydantic

from .object import ObjectKind


class BackgroundTaskState(str, enum.Enum):
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


class BackgroundTaskSpec(pydantic.BaseModel):
    pass


class BackgroundTaskStatus(pydantic.BaseModel):
    state: BackgroundTaskState


class BackgroundTask(pydantic.BaseModel):
    kind: ObjectKind = pydantic.Field(ObjectKind.background_task, const=True)
    metadata: BackgroundTaskMetadata
    spec: BackgroundTaskSpec
    status: BackgroundTaskStatus
