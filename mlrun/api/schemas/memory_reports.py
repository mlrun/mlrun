import typing

import pydantic


class MostCommonObjectTypesReport(pydantic.BaseModel):
    object_types: typing.List[typing.Tuple[str, int]]


class ObjectTypeReport(pydantic.BaseModel):
    object_type: str
    sample_size: int
    start_index: typing.Optional[int]
    max_depth: int
    object_report: typing.List[typing.Dict[str, typing.Any]]
