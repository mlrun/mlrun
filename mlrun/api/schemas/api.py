import enum

import pydantic


class APIState(pydantic.BaseModel):
    state: str


class WaitForChiefToReachOnlineStateFeatureFlag(str, enum.Enum):
    enabled = "enabled"
    disabled = "disabled"
