from typing import Optional, List

from pydantic import BaseModel, Extra
from .object import (
    ObjectMetadata,
    ObjectStatus,
    ObjectSpec,
    ObjectRecord,
    LabelRecord,
)


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


class FeatureSetSpec(ObjectSpec):
    entities: List[Entity]
    features: List[Feature]


class FeatureSet(BaseModel):
    kind: str = "FeatureSet"
    metadata: ObjectMetadata
    spec: FeatureSetSpec
    status: ObjectStatus


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


class FeatureSetRecord(ObjectRecord):
    entities: List[EntityRecord]
    features: List[FeatureRecord]

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


class FeatureVector(BaseModel):
    kind: str = "FeatureVector"
    metadata: ObjectMetadata
    spec: ObjectSpec
    status: ObjectStatus


class FeatureVectorRecord(ObjectRecord):
    pass


class FeatureVectorsOutput(BaseModel):
    feature_vectors: List[FeatureVector]
