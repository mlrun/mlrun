from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class Feature(BaseModel):
    name: str
    value_type: str
    labels: Optional[dict]


class Entity(BaseModel):
    name: str
    value_type: str
    labels: Optional[dict]


class FeatureSetMetadata(BaseModel):
    name: str
    tag: Optional[str]
    labels: Optional[dict]
    updated: Optional[datetime]
    uid: Optional[str]


class FeatureSetSpec(BaseModel):
    entities: List[Entity]
    features: List[Feature]


class FeatureSet(BaseModel):
    metadata: FeatureSetMetadata
    spec: FeatureSetSpec
    status: Optional[dict]


class FeatureSetUpdate(BaseModel):
    features: Optional[List[Feature]]
    entities: Optional[List[Feature]]
    status: Optional[dict]
    labels: Optional[dict]


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
    status: Optional[dict] = None

    class Config:
        orm_mode = True


class FeatureSetsOutput(BaseModel):
    feature_sets: List[FeatureSet]


class FeatureSetDigestSpec(BaseModel):
    entities: List[Entity]


class FeatureSetDigestOutput(BaseModel):
    metadata: FeatureSetMetadata
    spec: FeatureSetDigestSpec


class FeatureListOutput(BaseModel):
    feature: Feature
    feature_set_digests: List[FeatureSetDigestOutput]


class FeaturesOutput(BaseModel):
    features: List[FeatureListOutput]
