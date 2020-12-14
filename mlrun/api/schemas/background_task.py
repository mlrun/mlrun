import datetime
import enum
import typing

import pydantic

import mlrun.api.schemas.object


class BackgroundTaskState(str, enum.Enum):
    succeeded = "succeeded"
    failed = "failed"
    running = "running"


class BackgroundTaskMetadata(pydantic.BaseModel):
    name: str
    created: typing.Optional[datetime.datetime]
    updated: typing.Optional[datetime.datetime]


class BackgroundTaskSpec(pydantic.BaseModel):
    pass


class BackgroundTaskStatus(pydantic.BaseModel):
    state: BackgroundTaskState


class BackgroundTask(pydantic.BaseModel):
    kind: mlrun.api.schemas.object.ObjectKind = pydantic.Field(
        mlrun.api.schemas.object.ObjectKind.feature_set, const=True
    )
    metadata: BackgroundTaskMetadata
    spec: BackgroundTaskSpec
    status: BackgroundTaskStatus
