from typing import List, Optional

from pydantic import BaseModel, Extra, Field

from .auth import AuthorizationResourceTypes
from .object import (
    LabelRecord,
    ObjectKind,
    ObjectMetadata,
    ObjectRecord,
    ObjectSpec,
    ObjectStatus,
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
    entities: List[Entity] = []
    features: List[Feature] = []


class FeatureSet(BaseModel):
    kind: ObjectKind = Field(ObjectKind.feature_set, const=True)
    metadata: ObjectMetadata
    spec: FeatureSetSpec
    status: ObjectStatus

    @staticmethod
    def get_authorization_resource_type():
        return AuthorizationResourceTypes.feature_set


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


class FeatureSetsTagsOutput(BaseModel):
    tags: List[str] = []


class FeatureSetDigestSpec(BaseModel):
    entities: List[Entity]
    features: List[Feature]


class FeatureSetDigestOutput(BaseModel):
    metadata: ObjectMetadata
    spec: FeatureSetDigestSpec


class FeatureListOutput(BaseModel):
    feature: Feature
    feature_set_digest: FeatureSetDigestOutput


class FeaturesOutput(BaseModel):
    features: List[FeatureListOutput]


class EntityListOutput(BaseModel):
    entity: Entity
    feature_set_digest: FeatureSetDigestOutput


class EntitiesOutput(BaseModel):
    entities: List[EntityListOutput]


class FeatureVector(BaseModel):
    kind: ObjectKind = Field(ObjectKind.feature_vector, const=True)
    metadata: ObjectMetadata
    spec: ObjectSpec
    status: ObjectStatus

    @staticmethod
    def get_authorization_resource_type():
        return AuthorizationResourceTypes.feature_vector


class FeatureVectorRecord(ObjectRecord):
    pass


class FeatureVectorsOutput(BaseModel):
    feature_vectors: List[FeatureVector]


class FeatureVectorsTagsOutput(BaseModel):
    tags: List[str] = []


class DataSource(BaseModel):
    kind: str
    name: str
    path: str

    class Config:
        extra = Extra.allow


class DataTarget(BaseModel):
    kind: str
    name: str
    path: Optional[str]

    class Config:
        extra = Extra.allow


class Credentials(BaseModel):
    access_key: Optional[str]


class FeatureSetIngestInput(BaseModel):
    source: Optional[DataSource]
    targets: Optional[List[DataTarget]]
    infer_options: Optional[int]
    credentials: Credentials = Credentials()


class FeatureSetIngestOutput(BaseModel):
    feature_set: FeatureSet
    run_object: dict
