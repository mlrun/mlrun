import enum
import typing

import pydantic


class ListRuntimeResourcesGroupByField(str, enum.Enum):
    job = "job"


class RuntimeResourcesOutput(pydantic.BaseModel):
    crd_resources: typing.List[typing.Dict]
    pod_resources: typing.List[typing.Dict]

    class Config:
        extra = pydantic.Extra.allow


# project name -> job uid -> runtime resources
GroupedRuntimeResourcesOutput = typing.Dict[
    str, typing.Dict[str, RuntimeResourcesOutput]
]
