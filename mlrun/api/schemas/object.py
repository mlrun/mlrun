from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Extra


class ObjectMetadata(BaseModel):
    name: str
    project: Optional[str]
    tag: Optional[str]
    labels: Optional[dict]
    updated: Optional[datetime]
    uid: Optional[str]

    class Config:
        extra = Extra.allow


class ObjectKind(str, Enum):
    feature_set = "FeatureSet"
