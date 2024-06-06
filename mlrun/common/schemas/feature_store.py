# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Optional

from pydantic import BaseModel, Extra, Field

from .auth import AuthorizationResourceTypes, Credentials
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
    labels: Optional[dict] = {}

    class Config:
        extra = Extra.allow


class Entity(BaseModel):
    name: str
    value_type: str
    labels: Optional[dict] = {}

    class Config:
        extra = Extra.allow


class QualifiedEntity(BaseModel):
    name: str
    value_type: str
    project: str
    feature_set_name: str
    feature_set_tag: Optional[str]
    labels: Optional[dict] = {}

    class Config:
        extra = Extra.allow


class FeatureSetSpec(ObjectSpec):
    entities: list[Entity] = []
    features: list[Feature] = []
    engine: Optional[str] = Field(default="storey")


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
    labels: list[LabelRecord]

    class Config:
        orm_mode = True


class FeatureRecord(BaseModel):
    name: str
    value_type: str
    labels: list[LabelRecord]

    class Config:
        orm_mode = True


class FeatureSetRecord(ObjectRecord):
    entities: list[EntityRecord]
    features: list[FeatureRecord]

    class Config:
        orm_mode = True


class FeatureSetsOutput(BaseModel):
    feature_sets: list[FeatureSet]


class FeatureSetsTagsOutput(BaseModel):
    tags: list[str] = []


class FeatureSetDigestSpec(BaseModel):
    entities: list[Entity]
    features: list[Feature]


class FeatureSetDigestOutput(BaseModel):
    metadata: ObjectMetadata
    spec: FeatureSetDigestSpec

    class Config:
        copy_on_model_validation = False


class FeatureSetDigestSpecV2(BaseModel):
    entities: list[Entity]


class FeatureSetDigestOutputV2(BaseModel):
    metadata: ObjectMetadata
    spec: FeatureSetDigestSpecV2


class FeatureListOutput(BaseModel):
    feature: Feature
    feature_set_digest: FeatureSetDigestOutput


class FeaturesOutput(BaseModel):
    features: list[FeatureListOutput]


class EntityListOutput(BaseModel):
    entity: Entity
    feature_set_digest: FeatureSetDigestOutput


class EntitiesOutputV2(BaseModel):
    entities: list[QualifiedEntity]
    feature_set_digests: list[FeatureSetDigestOutputV2]


class EntitiesOutput(BaseModel):
    entities: list[EntityListOutput]


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
    feature_vectors: list[FeatureVector]


class FeatureVectorsTagsOutput(BaseModel):
    tags: list[str] = []


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


class FeatureSetIngestInput(BaseModel):
    source: Optional[DataSource]
    targets: Optional[list[DataTarget]]
    infer_options: Optional[int]
    credentials: Credentials = Credentials()


class FeatureSetIngestOutput(BaseModel):
    feature_set: FeatureSet
    run_object: dict
