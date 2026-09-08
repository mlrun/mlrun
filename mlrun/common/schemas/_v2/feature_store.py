# Copyright 2026 Iguazio
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


import typing

import pydantic

from .._shared.auth import AuthorizationResourceTypes
from .._shared.object import ObjectKind
from .auth import Credentials
from .object import (
    LabelRecord,
    ObjectMetadata,
    ObjectRecord,
    ObjectSpec,
    ObjectStatus,
)


class FeatureStoreBaseModel(pydantic.BaseModel):
    """
    Intermediate base class for the feature-store schemas.

    In pydantic 1 this disabled copy-on-validation (``copy_on_model_validation = "none"``) so nested
    models were kept by reference; pydantic 2 no longer copies models on validation, so no config
    override is required here.
    """


class Feature(FeatureStoreBaseModel):
    name: str
    value_type: str
    labels: dict | None = {}

    model_config = pydantic.ConfigDict(extra="allow")


class Entity(FeatureStoreBaseModel):
    name: str
    value_type: str
    labels: dict | None = {}

    model_config = pydantic.ConfigDict(extra="allow")


class FeatureSetSpec(ObjectSpec):
    entities: list[Entity] = []
    features: list[Feature] = []
    engine: str | None = pydantic.Field(default="storey")


class FeatureSet(FeatureStoreBaseModel):
    kind: typing.Literal[ObjectKind.feature_set] = ObjectKind.feature_set
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

    model_config = pydantic.ConfigDict(from_attributes=True)


class FeatureRecord(FeatureStoreBaseModel):
    name: str
    value_type: str
    labels: list[LabelRecord]

    model_config = pydantic.ConfigDict(from_attributes=True)


class FeatureSetRecord(ObjectRecord):
    entities: list[EntityRecord]
    features: list[FeatureRecord]

    model_config = pydantic.ConfigDict(from_attributes=True)


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
    kind: typing.Literal[ObjectKind.feature_vector] = ObjectKind.feature_vector
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

    model_config = pydantic.ConfigDict(extra="allow")


class DataTarget(FeatureStoreBaseModel):
    kind: str
    name: str
    path: str | None = None

    model_config = pydantic.ConfigDict(extra="allow")


class FeatureSetIngestInput(FeatureStoreBaseModel):
    source: DataSource | None = None
    targets: list[DataTarget] | None = None
    infer_options: int | None = None
    credentials: Credentials = Credentials()


class FeatureSetIngestOutput(FeatureStoreBaseModel):
    feature_set: FeatureSet
    run_object: dict
