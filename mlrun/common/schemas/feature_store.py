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

import pydantic.v1

from .auth import AuthorizationResourceTypes, Credentials
from .object import (
    LabelRecord,
    ObjectKind,
    ObjectMetadata,
    ObjectRecord,
    ObjectSpec,
    ObjectStatus,
)


class FeatureStoreBaseModel(pydantic.v1.BaseModel):
    """
    Intermediate base class, in order to override pydantic's configuration, as per
    https://docs.pydantic.dev/1.10/usage/model_config/#change-behaviour-globally
    """

    class Config:
        copy_on_model_validation = "none"


class Feature(FeatureStoreBaseModel):
    name: str
    value_type: str
    labels: Optional[dict] = {}

    class Config:
        extra = pydantic.v1.Extra.allow


class Entity(FeatureStoreBaseModel):
    name: str
    value_type: str
    labels: Optional[dict] = {}

    class Config:
        extra = pydantic.v1.Extra.allow


class FeatureSetSpec(ObjectSpec):
    entities: list[Entity] = []
    features: list[Feature] = []
    engine: Optional[str] = pydantic.v1.Field(default="storey")


class FeatureSet(FeatureStoreBaseModel):
    kind: ObjectKind = pydantic.v1.Field(ObjectKind.feature_set, const=True)
    metadata: ObjectMetadata
    spec: FeatureSetSpec
    status: ObjectStatus

    @staticmethod
    def get_authorization_resource_type():
        return AuthorizationResourceTypes.feature_set


class EntityRecord(FeatureStoreBaseModel):
    name: str
    value_type: str
    labels: list[LabelRecord]

    class Config:
        orm_mode = True


class FeatureRecord(FeatureStoreBaseModel):
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


class FeatureSetsOutput(FeatureStoreBaseModel):
    feature_sets: list[FeatureSet]


class FeatureSetsTagsOutput(FeatureStoreBaseModel):
    tags: list[str] = []


class FeatureSetDigestSpec(FeatureStoreBaseModel):
    entities: list[Entity]
    features: list[Feature]


class FeatureSetDigestOutput(FeatureStoreBaseModel):
    metadata: ObjectMetadata
    spec: FeatureSetDigestSpec


class FeatureSetDigestSpecV2(FeatureStoreBaseModel):
    entities: list[Entity]


class FeatureSetDigestOutputV2(FeatureStoreBaseModel):
    feature_set_index: int
    metadata: ObjectMetadata
    spec: FeatureSetDigestSpecV2


class FeatureListOutput(FeatureStoreBaseModel):
    feature: Feature
    feature_set_digest: FeatureSetDigestOutput


class FeaturesOutput(FeatureStoreBaseModel):
    features: list[FeatureListOutput]


class FeaturesOutputV2(FeatureStoreBaseModel):
    features: list[Feature]
    feature_set_digests: list[FeatureSetDigestOutputV2]


class EntityListOutput(FeatureStoreBaseModel):
    entity: Entity
    feature_set_digest: FeatureSetDigestOutput


class EntitiesOutputV2(FeatureStoreBaseModel):
    entities: list[Entity]
    feature_set_digests: list[FeatureSetDigestOutputV2]


class EntitiesOutput(FeatureStoreBaseModel):
    entities: list[EntityListOutput]


class FeatureVector(FeatureStoreBaseModel):
    kind: ObjectKind = pydantic.v1.Field(ObjectKind.feature_vector, const=True)
    metadata: ObjectMetadata
    spec: ObjectSpec
    status: ObjectStatus

    @staticmethod
    def get_authorization_resource_type():
        return AuthorizationResourceTypes.feature_vector


class FeatureVectorRecord(ObjectRecord):
    pass


class FeatureVectorsOutput(FeatureStoreBaseModel):
    feature_vectors: list[FeatureVector]


class FeatureVectorsTagsOutput(FeatureStoreBaseModel):
    tags: list[str] = []


class DataSource(FeatureStoreBaseModel):
    kind: str
    name: str
    path: str

    class Config:
        extra = pydantic.v1.Extra.allow


class DataTarget(FeatureStoreBaseModel):
    kind: str
    name: str
    path: Optional[str]

    class Config:
        extra = pydantic.v1.Extra.allow


class FeatureSetIngestInput(FeatureStoreBaseModel):
    source: Optional[DataSource]
    targets: Optional[list[DataTarget]]
    infer_options: Optional[int]
    credentials: Credentials = Credentials()


class FeatureSetIngestOutput(FeatureStoreBaseModel):
    feature_set: FeatureSet
    run_object: dict
