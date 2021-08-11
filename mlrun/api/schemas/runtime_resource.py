import enum
import typing

import pydantic


class ListRuntimeResourcesGroupByField(str, enum.Enum):
    job = "job"


class RuntimeResource(pydantic.BaseModel):
    name: str
    labels: typing.Dict[str, str] = {}
    status: typing.Optional[typing.Dict]


class RuntimeResources(pydantic.BaseModel):
    crd_resources: typing.List[RuntimeResource]
    pod_resources: typing.List[RuntimeResource]
    # only for dask runtime
    service_resources: typing.Optional[typing.List[RuntimeResource]]

    class Config:
        extra = pydantic.Extra.allow


class SpecificKindRuntimeResources(pydantic.BaseModel):
    kind: str
    resources = RuntimeResources


RuntimeResourcesOutput = typing.List[SpecificKindRuntimeResources]


# project name -> job uid -> runtime resources
GroupedRuntimeResourcesOutput = typing.Dict[str, typing.Dict[str, RuntimeResources]]
