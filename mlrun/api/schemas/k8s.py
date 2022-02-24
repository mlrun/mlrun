import typing

import pydantic


class ResourceSpec(pydantic.BaseModel):
    cpu: typing.Optional[str]
    memory: typing.Optional[str]
    gpu: typing.Optional[str]


class Resources(pydantic.BaseModel):
    requests: ResourceSpec = ResourceSpec()
    limits: ResourceSpec = ResourceSpec()
