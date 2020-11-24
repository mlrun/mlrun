from datetime import datetime
from enum import Enum
from typing import Optional

import mergedeep
from pydantic import BaseModel, Extra

import mlrun.errors


class ObjectMetadata(BaseModel):
    name: str
    tag: Optional[str]
    labels: Optional[dict]
    updated: Optional[datetime]
    uid: Optional[str]

    class Config:
        extra = Extra.allow


class PatchMode(str, Enum):
    replace = "replace"
    additive = "additive"

    def to_mergedeep_strategy(self) -> mergedeep.Strategy:
        if self.value == PatchMode.replace:
            return mergedeep.Strategy.REPLACE
        elif self.value == PatchMode.additive:
            return mergedeep.Strategy.ADDITIVE
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(f"Unknown patch mode: {self.value}")
