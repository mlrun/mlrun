# Copyright 2024 Iguazio
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

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

import mlrun.common.schemas
import mlrun.errors
import mlrun.feature_store
import server.api.api.endpoints.feature_store
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.singletons.project_member
from mlrun.common.schemas.feature_store import (
    FeatureSetDigestOutputV2,
    FeatureSetDigestSpecV2,
    QualifiedEntity,
    QualifiedFeature,
)
from server.api.api import deps

router = APIRouter(prefix="/v2/projects/{project}")


def _dedup_feature_set(
    feature_set_digest, feature_set_digest_id_to_index, feature_set_digests_v2
):
    # dedup feature set list
    # we can rely on the object ID because SQLAlchemy already avoids duplication at the object
    # level, and the conversion from "model" to "schema" retains this property
    feature_set_digest_obj_id = id(feature_set_digest)
    feature_set_index = feature_set_digest_id_to_index.get(
        feature_set_digest_obj_id, None
    )
    if feature_set_index is None:
        feature_set_index = len(feature_set_digest_id_to_index)
        feature_set_digest_id_to_index[feature_set_digest_obj_id] = feature_set_index
        feature_set_digests_v2.append(
            FeatureSetDigestOutputV2(
                feature_set_index=feature_set_index,
                metadata=feature_set_digest.metadata,
                spec=FeatureSetDigestSpecV2(
                    entities=feature_set_digest.spec.entities,
                ),
            )
        )
    return feature_set_index


@router.get("/entities", response_model=mlrun.common.schemas.EntitiesOutputV2)
async def list_entities(
    project: str,
    name: str = None,
    tag: str = None,
    labels: list[str] = Query(None, alias="label"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    entities = await server.api.api.endpoints.feature_store.list_entities(
        project,
        name,
        tag,
        labels,
        auth_info,
        db_session,
    )

    entities_v2: list[QualifiedEntity] = []
    feature_set_digests_v2: list[FeatureSetDigestOutputV2] = []
    feature_set_digest_id_to_index: dict[int, int] = {}

    for entity_v1 in entities.entities:
        feature_set_digest = entity_v1.feature_set_digest

        feature_set_index = _dedup_feature_set(
            feature_set_digest, feature_set_digest_id_to_index, feature_set_digests_v2
        )

        entity = entity_v1.entity
        entities_v2.append(
            QualifiedEntity(
                name=entity.name,
                value_type=entity.value_type,
                feature_set_index=feature_set_index,
                labels=entity.labels,
            )
        )

    return mlrun.common.schemas.EntitiesOutputV2(
        entities=entities_v2, feature_set_digests=feature_set_digests_v2
    )


@router.get("/features", response_model=mlrun.common.schemas.FeaturesOutputV2)
async def list_features(
    project: str,
    name: str = None,
    tag: str = None,
    entities: list[str] = Query(None, alias="entity"),
    labels: list[str] = Query(None, alias="label"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    features = await server.api.api.endpoints.feature_store.list_features(
        project=project,
        name=name,
        tag=tag,
        entities=entities,
        labels=labels,
        auth_info=auth_info,
        db_session=db_session,
    )

    features_v2: list[QualifiedFeature] = []
    feature_set_digests_v2: list[FeatureSetDigestOutputV2] = []
    feature_set_digest_id_to_index: dict[int, int] = {}

    for feature_v1 in features.features:
        feature_set_digest = feature_v1.feature_set_digest

        feature_set_index = _dedup_feature_set(
            feature_set_digest, feature_set_digest_id_to_index, feature_set_digests_v2
        )

        feature = feature_v1.feature
        features_v2.append(
            QualifiedFeature(
                name=feature.name,
                value_type=feature.value_type,
                feature_set_index=feature_set_index,
                labels=feature.labels,
            )
        )

    return mlrun.common.schemas.FeaturesOutputV2(
        features=features_v2, feature_set_digests=feature_set_digests_v2
    )
