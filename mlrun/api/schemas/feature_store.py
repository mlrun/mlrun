from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Extra
from .object import ObjectMetadata


class Feature(BaseModel):
    name: str
    value_type: str
    labels: Optional[dict]

    class Config:
        extra = Extra.allow


class Entity(BaseModel):
    name: str
    value_type: str
    labels: Optional[dict]

    class Config:
        extra = Extra.allow


class FeatureSetSpec(BaseModel):
    entities: List[Entity]
    features: List[Feature]

    class Config:
        extra = Extra.allow


class FeatureSetStatus(BaseModel):
    state: Optional[str]

    class Config:
        extra = Extra.allow


class FeatureSet(BaseModel):
    metadata: ObjectMetadata
    spec: FeatureSetSpec
    status: FeatureSetStatus


class LabelRecord(BaseModel):
    id: int
    name: str
    value: str

    class Config:
        orm_mode = True


class EntityRecord(BaseModel):
    name: str
    value_type: str
    labels: List[LabelRecord]

    class Config:
        orm_mode = True


class FeatureRecord(BaseModel):
    name: str
    value_type: str
    labels: List[LabelRecord]

    class Config:
        orm_mode = True


class FeatureSetRecord(BaseModel):
    id: int
    name: str
    project: str
    uid: str
    updated: Optional[datetime] = None
    entities: List[EntityRecord]
    features: List[FeatureRecord]
    labels: List[LabelRecord]
    # state is extracted from the full status dict to enable queries
    state: Optional[str] = None
    full_object: Optional[dict] = None

    class Config:
        orm_mode = True


class FeatureSetsOutput(BaseModel):
    feature_sets: List[FeatureSet]


class FeatureSetDigestSpec(BaseModel):
    entities: List[Entity]


class FeatureSetDigestOutput(BaseModel):
    metadata: ObjectMetadata
    spec: FeatureSetDigestSpec


class FeatureListOutput(BaseModel):
    feature: Feature
    feature_set_digest: FeatureSetDigestOutput


class FeaturesOutput(BaseModel):
    features: List[FeatureListOutput]
