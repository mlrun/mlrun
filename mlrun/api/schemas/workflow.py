import typing

import pydantic

from .schedule import ScheduleCronTrigger


class WorkflowSpec(pydantic.BaseModel):
    name: typing.Optional[str] = None
    engine: typing.Optional[str] = None
    code: typing.Optional[str] = None
    path: typing.Optional[str] = None
    args: typing.Optional[dict] = None
    handler: typing.Optional[str] = None
    ttl: typing.Optional[int] = None
    args_schema: typing.Optional[list] = None
    schedule: typing.Union[str, ScheduleCronTrigger] = None
    run_local: typing.Optional[bool] = None
