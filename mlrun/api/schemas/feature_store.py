from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


# Shared properties
class Feature(BaseModel):
    name: str
    description: Optional[str] = None
    value_type: str

    class Config:
        orm_mode = True


class FeatureSet(BaseModel):
    name: str
    description: Optional[str] = None
    updated: Optional[datetime] = None
    features: List[Feature]
    entities: List[Feature]
    status: Optional[dict] = None


class FeatureSetInDB(FeatureSet):
    project: str
    id: int = None

    class Config:
        orm_mode = True


