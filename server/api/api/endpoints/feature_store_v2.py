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
)
from server.api.api import deps

router = APIRouter(prefix="/v2/projects/{project}")


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
    feature_set_digest_id_set: set[int] = set()

    for entity_v1 in entities.entities:
        entity = entity_v1.entity
        feature_set_digest = entity_v1.feature_set_digest
        entities_v2.append(
            QualifiedEntity(
                name=entity.name,
                value_type=entity.value_type,
                project=project,
                feature_set_name=feature_set_digest.metadata.name,
                feature_set_tag=feature_set_digest.metadata.tag,
                labels=entity.labels,
            )
        )
        # dedup feature set list
        # TODO: we should be able to rely on the object ID because SQLAlchemy already avoids duplication at the object
        # level, but the conversion from "model" to "schema" messes that up and duplicates the objects
        # feature_set_digest_obj_id = id(feature_set_digest)
        feature_set_digest_obj_id = hash(
            (feature_set_digest.metadata.name, feature_set_digest.metadata.tag)
        )
        if feature_set_digest_obj_id not in feature_set_digest_id_set:
            feature_set_digest_id_set.add(feature_set_digest_obj_id)
            feature_set_digests_v2.append(
                FeatureSetDigestOutputV2(
                    metadata=feature_set_digest.metadata,
                    spec=FeatureSetDigestSpecV2(
                        entities=feature_set_digest.spec.entities,
                    ),
                )
            )

    return mlrun.common.schemas.EntitiesOutputV2(
        entities=entities_v2, feature_set_digests=feature_set_digests_v2
    )
