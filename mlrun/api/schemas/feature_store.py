from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class Feature(BaseModel):
    name: str
    value_type: str

    class Config:
        orm_mode = True


class FeatureSetMetaData(BaseModel):
    name: str
    labels: Optional[dict]


class FeatureSetSpec(BaseModel):
    entities: List[Feature]
    features: List[Feature]


class FeatureSet(BaseModel):
    metadata: FeatureSetMetaData
    spec: FeatureSetSpec
    status: Optional[dict]


class FeatureSetUpdate(BaseModel):
    features: Optional[List[Feature]]
    entities: Optional[List[Feature]]
    status: Optional[dict]
    labels: Optional[dict]


# state is extracted from the full status dict to enable queries
class FeatureSetRecord(BaseModel):
    id: int
    name: str
    project: str
    updated: Optional[datetime] = None
    entities: List[Feature]
    features: List[Feature]
    state: Optional[str] = None
    status: Optional[dict] = None

    class Config:
        orm_mode = True


