import enum
import typing

import pydantic


class ClusterizationSpec(pydantic.BaseModel):
    chief_api_state: str
    chief_version: typing.Optional[str]


class WaitForChiefToReachOnlineStateFeatureFlag(str, enum.Enum):
    enabled = "enabled"
    disabled = "disabled"
