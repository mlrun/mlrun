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
import asyncio
from http import HTTPStatus
from typing import Optional

from fastapi import APIRouter, Depends, Header, Query, Response
from fastapi.concurrency import run_in_threadpool
from mlrun_pipelines.mounts import v3io_cred
from sqlalchemy.orm import Session

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.errors
import mlrun.feature_store
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.singletons.project_member
from mlrun.data_types import InferOptions
from mlrun.datastore.targets import get_default_prefix_for_target
from mlrun.feature_store.api import RunConfig, ingest
from mlrun.model import DataSource, DataTargetBase
from server.api.api import deps
from server.api.api.utils import log_and_raise, parse_reference

router = APIRouter(prefix="/projects/{project}")


@router.post("/feature-sets", response_model=mlrun.common.schemas.FeatureSet)
async def create_feature_set(
    project: str,
    feature_set: mlrun.common.schemas.FeatureSet,
    versioned: bool = True,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_set,
        project,
        feature_set.metadata.name,
        mlrun.common.schemas.AuthorizationAction.create,
        auth_info,
    )
    feature_set_uid = await run_in_threadpool(
        server.api.crud.FeatureStore().create_feature_set,
        db_session,
        project,
        feature_set,
        versioned,
    )

    return await run_in_threadpool(
        server.api.crud.FeatureStore().get_feature_set,
        db_session,
        project,
        feature_set.metadata.name,
        feature_set.metadata.tag or "latest",
        feature_set_uid,
    )


@router.put(
    "/feature-sets/{name}/references/{reference}",
    response_model=mlrun.common.schemas.FeatureSet,
)
async def store_feature_set(
    project: str,
    name: str,
    reference: str,
    feature_set: mlrun.common.schemas.FeatureSet,
    versioned: bool = True,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_set,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    tag, uid = parse_reference(reference)
    uid = await run_in_threadpool(
        server.api.crud.FeatureStore().store_feature_set,
        db_session,
        project,
        name,
        feature_set,
        tag,
        uid,
        versioned,
    )
    return await run_in_threadpool(
        server.api.crud.FeatureStore().get_feature_set,
        db_session,
        project,
        feature_set.metadata.name,
        tag,
        uid,
    )


@router.patch("/feature-sets/{name}/references/{reference}")
async def patch_feature_set(
    project: str,
    name: str,
    feature_set_update: dict,
    reference: str,
    patch_mode: mlrun.common.schemas.PatchMode = Header(
        mlrun.common.schemas.PatchMode.replace,
        alias=mlrun.common.schemas.HeaderNames.patch_mode,
    ),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_set,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.update,
        auth_info,
    )
    tag, uid = parse_reference(reference)
    await run_in_threadpool(
        server.api.crud.FeatureStore().patch_feature_set,
        db_session,
        project,
        name,
        feature_set_update,
        tag,
        uid,
        patch_mode,
    )
    return Response(status_code=HTTPStatus.OK.value)


@router.get(
    "/feature-sets/{name}/references/{reference}",
    response_model=mlrun.common.schemas.FeatureSet,
)
async def get_feature_set(
    project: str,
    name: str,
    reference: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(reference)
    feature_set = await run_in_threadpool(
        server.api.crud.FeatureStore().get_feature_set,
        db_session,
        project,
        name,
        tag,
        uid,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_set,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return feature_set


@router.delete("/feature-sets/{name}")
@router.delete("/feature-sets/{name}/references/{reference}")
async def delete_feature_set(
    project: str,
    name: str,
    reference: str = None,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_set,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    tag = uid = None
    if reference:
        tag, uid = parse_reference(reference)
    await run_in_threadpool(
        server.api.crud.FeatureStore().delete_feature_set,
        db_session,
        project,
        name,
        tag,
        uid,
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get(
    "/feature-sets",
    response_model=mlrun.common.schemas.FeatureSetsOutput,
)
async def list_feature_sets(
    project: str,
    name: str = None,
    state: str = None,
    tag: str = None,
    entities: list[str] = Query(None, alias="entity"),
    features: list[str] = Query(None, alias="feature"),
    labels: list[str] = Query(None, alias="label"),
    partition_by: mlrun.common.schemas.FeatureStorePartitionByField = Query(
        None, alias="partition-by"
    ),
    rows_per_partition: int = Query(1, alias="rows-per-partition", gt=0),
    partition_sort_by: mlrun.common.schemas.SortField = Query(
        None, alias="partition-sort-by"
    ),
    partition_order: mlrun.common.schemas.OrderType = Query(
        mlrun.common.schemas.OrderType.desc, alias="partition-order"
    ),
    format_: str = Query(mlrun.common.formatters.FeatureSetFormat.full, alias="format"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    feature_sets = await run_in_threadpool(
        server.api.crud.FeatureStore().list_feature_sets,
        db_session,
        project,
        name,
        tag,
        state,
        entities,
        features,
        labels,
        partition_by,
        rows_per_partition,
        partition_sort_by,
        partition_order,
        format_,
    )
    feature_sets = await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_set,
        feature_sets.feature_sets,
        lambda feature_set: (
            feature_set.metadata.project,
            feature_set.metadata.name,
        ),
        auth_info,
    )
    return mlrun.common.schemas.FeatureSetsOutput(feature_sets=feature_sets)


@router.get(
    "/feature-sets/{name}/tags",
    response_model=mlrun.common.schemas.FeatureSetsTagsOutput,
)
async def list_feature_sets_tags(
    project: str,
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    if name != "*":
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Listing a specific feature set tags is not supported, set name to *"
        )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    tag_tuples = await run_in_threadpool(
        server.api.crud.FeatureStore().list_feature_sets_tags,
        db_session,
        project,
    )
    feature_set_name_to_tag = {tag_tuple[1]: tag_tuple[2] for tag_tuple in tag_tuples}
    auth_verifier = server.api.utils.auth.verifier.AuthVerifier()
    allowed_feature_set_names = (
        await auth_verifier.filter_project_resources_by_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.feature_set,
            list(feature_set_name_to_tag.keys()),
            lambda feature_set_name: (
                project,
                feature_set_name,
            ),
            auth_info,
        )
    )
    tags = {
        tag_tuple[2]
        for tag_tuple in tag_tuples
        if tag_tuple[1] in allowed_feature_set_names
    }
    return mlrun.common.schemas.FeatureSetsTagsOutput(tags=list(tags))


def _has_v3io_path(data_source, data_targets, feature_set):
    paths = []

    # If no data targets received, then use targets from the feature-set spec. In case it's empty as well, use
    # default targets (by calling set_targets())
    if not data_targets:
        if not feature_set.spec.targets:
            feature_set.set_targets()
        data_targets = feature_set.spec.targets

    if data_targets:
        for target in data_targets:
            # If the target does not have a path (i.e. default target), then retrieve the default path from config.
            paths.append(target.path or get_default_prefix_for_target(target.kind))

    source = data_source or feature_set.spec.source
    if source:
        paths.append(source.path)

    return any(
        path and (path.startswith("v3io://") or path.startswith("v3ios://"))
        for path in paths
    )


@router.post(
    "/feature-sets/{name}/references/{reference}/ingest",
    response_model=mlrun.common.schemas.FeatureSetIngestOutput,
    status_code=HTTPStatus.ACCEPTED.value,
)
async def ingest_feature_set(
    project: str,
    name: str,
    reference: str,
    ingest_parameters: Optional[
        mlrun.common.schemas.FeatureSetIngestInput
    ] = mlrun.common.schemas.FeatureSetIngestInput(),
    username: str = Header(None, alias="x-remote-user"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    """
    This endpoint is being called only through the UI, this is mainly for enrichment of the feature set
    that already being happen on client side
    """
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_set,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.update,
        auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.run,
        project,
        "",
        mlrun.common.schemas.AuthorizationAction.create,
        auth_info,
    )
    data_source = data_targets = None
    if ingest_parameters.source:
        data_source = DataSource.from_dict(ingest_parameters.source.dict())
    if data_source.schedule:
        await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.schedule,
            project,
            "",
            mlrun.common.schemas.AuthorizationAction.create,
            auth_info,
        )
    tag, uid = parse_reference(reference)
    feature_set_record = await run_in_threadpool(
        server.api.crud.FeatureStore().get_feature_set,
        db_session,
        project,
        name,
        tag,
        uid,
    )
    feature_set = mlrun.feature_store.FeatureSet.from_dict(feature_set_record.dict())
    if feature_set.spec.function and feature_set.spec.function.function_object:
        function = feature_set.spec.function.function_object
        await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.function,
            function.metadata.project,
            function.metadata.name,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )

    # Set the run db instance with the current db session
    await run_in_threadpool(
        feature_set._override_run_db,
        server.api.api.utils.get_run_db_instance(db_session),
    )
    if ingest_parameters.targets:
        data_targets = [
            DataTargetBase.from_dict(data_target.dict())
            for data_target in ingest_parameters.targets
        ]

    run_config = RunConfig(
        owner=username,
        credentials=mlrun.model.Credentials(ingest_parameters.credentials.access_key),
        # setting auth_info to indicate that we are running on server side
        auth_info=auth_info,
    )

    # Try to deduce whether the ingest job will need v3io mount, by analyzing the paths to the source and
    # targets. If it needs it, apply v3io mount to the run_config. Note that the access-key and username are
    # user-context parameters, we cannot use the api context.
    if _has_v3io_path(data_source, data_targets, feature_set):
        access_key = auth_info.data_session

        if not access_key or not username:
            log_and_raise(
                HTTPStatus.BAD_REQUEST.value,
                reason="Request needs v3io access key and username in header",
            )
        run_config = run_config.apply(v3io_cred(access_key=access_key, user=username))

    infer_options = ingest_parameters.infer_options or InferOptions.default()

    run_params = await run_in_threadpool(
        ingest,
        feature_set,
        data_source,
        data_targets,
        infer_options=infer_options,
        return_df=False,
        run_config=run_config,
    )
    # ingest may modify the feature-set contents, so returning the updated feature-set.
    result_feature_set = mlrun.common.schemas.FeatureSet(**feature_set.to_dict())
    return mlrun.common.schemas.FeatureSetIngestOutput(
        feature_set=result_feature_set, run_object=run_params.to_dict()
    )


# TODO: Remove in 1.9.0
@router.get(
    "/features",
    response_model=mlrun.common.schemas.FeaturesOutput,
    deprecated=True,
    description="/features v1 is deprecated in 1.7.0 and will be removed in 1.9.0. Use v2 instead.",
)
async def list_features(
    project: str,
    name: str = None,
    tag: str = None,
    entities: list[str] = Query(None, alias="entity"),
    labels: list[str] = Query(None, alias="label"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    features = await run_in_threadpool(
        server.api.crud.FeatureStore().list_features,
        db_session,
        project,
        name,
        tag,
        entities,
        labels,
    )
    features = await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature,
        features.features,
        lambda feature_list_output: (
            feature_list_output.feature_set_digest.metadata.project,
            feature_list_output.feature.name,
        ),
        auth_info,
    )
    return mlrun.common.schemas.FeaturesOutput(features=features)


# TODO: Remove in 1.9.0
@router.get(
    "/entities",
    response_model=mlrun.common.schemas.EntitiesOutput,
    deprecated=True,
    description="/entities v1 is deprecated in 1.7.0 and will be removed in 1.9.0. Use v2 instead.",
)
async def list_entities(
    project: str,
    name: str = None,
    tag: str = None,
    labels: list[str] = Query(None, alias="label"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    entities = await run_in_threadpool(
        server.api.crud.FeatureStore().list_entities,
        db_session,
        project,
        name,
        tag,
        labels,
    )
    entities = await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.entity,
        entities.entities,
        lambda entity_list_output: (
            entity_list_output.feature_set_digest.metadata.project,
            entity_list_output.entity.name,
        ),
        auth_info,
    )
    return mlrun.common.schemas.EntitiesOutput(entities=entities)


@router.post(
    "/feature-vectors",
    response_model=mlrun.common.schemas.FeatureVector,
)
async def create_feature_vector(
    project: str,
    feature_vector: mlrun.common.schemas.FeatureVector,
    versioned: bool = True,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_vector,
        project,
        feature_vector.metadata.name,
        mlrun.common.schemas.AuthorizationAction.create,
        auth_info,
    )
    await _verify_feature_vector_features_permissions(
        auth_info, project, feature_vector.dict()
    )
    feature_vector_uid = await run_in_threadpool(
        server.api.crud.FeatureStore().create_feature_vector,
        db_session,
        project,
        feature_vector,
        versioned,
    )

    return await run_in_threadpool(
        server.api.crud.FeatureStore().get_feature_vector,
        db_session,
        project,
        feature_vector.metadata.name,
        feature_vector.metadata.tag or "latest",
        feature_vector_uid,
    )


@router.get(
    "/feature-vectors/{name}/references/{reference}",
    response_model=mlrun.common.schemas.FeatureVector,
)
async def get_feature_vector(
    project: str,
    name: str,
    reference: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(reference)
    feature_vector = await run_in_threadpool(
        server.api.crud.FeatureStore().get_feature_vector,
        db_session,
        project,
        name,
        tag,
        uid,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_vector,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    await _verify_feature_vector_features_permissions(
        auth_info, project, feature_vector.dict()
    )
    return feature_vector


@router.get(
    "/feature-vectors",
    response_model=mlrun.common.schemas.FeatureVectorsOutput,
)
async def list_feature_vectors(
    project: str,
    name: str = None,
    state: str = None,
    tag: str = None,
    labels: list[str] = Query(None, alias="label"),
    partition_by: mlrun.common.schemas.FeatureStorePartitionByField = Query(
        None, alias="partition-by"
    ),
    rows_per_partition: int = Query(1, alias="rows-per-partition", gt=0),
    partition_sort_by: mlrun.common.schemas.SortField = Query(
        None, alias="partition-sort-by"
    ),
    partition_order: mlrun.common.schemas.OrderType = Query(
        mlrun.common.schemas.OrderType.desc, alias="partition-order"
    ),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    feature_vectors = await run_in_threadpool(
        server.api.crud.FeatureStore().list_feature_vectors,
        db_session,
        project,
        name,
        tag,
        state,
        labels,
        partition_by,
        rows_per_partition,
        partition_sort_by,
        partition_order,
    )
    feature_vectors = await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_vector,
        feature_vectors.feature_vectors,
        lambda feature_vector: (
            feature_vector.metadata.project,
            feature_vector.metadata.name,
        ),
        auth_info,
    )
    await asyncio.gather(
        *[
            _verify_feature_vector_features_permissions(auth_info, project, fv.dict())
            for fv in feature_vectors
        ]
    )
    return mlrun.common.schemas.FeatureVectorsOutput(feature_vectors=feature_vectors)


@router.get(
    "/feature-vectors/{name}/tags",
    response_model=mlrun.common.schemas.FeatureVectorsTagsOutput,
)
async def list_feature_vectors_tags(
    project: str,
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    if name != "*":
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Listing a specific feature vector tags is not supported, set name to *"
        )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    tag_tuples = await run_in_threadpool(
        server.api.crud.FeatureStore().list_feature_vectors_tags,
        db_session,
        project,
    )
    feature_vector_name_to_tag = {
        tag_tuple[1]: tag_tuple[2] for tag_tuple in tag_tuples
    }
    auth_verifier = server.api.utils.auth.verifier.AuthVerifier()
    allowed_feature_vector_names = (
        await auth_verifier.filter_project_resources_by_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.feature_vector,
            list(feature_vector_name_to_tag.keys()),
            lambda feature_vector_name: (
                project,
                feature_vector_name,
            ),
            auth_info,
        )
    )
    tags = {
        tag_tuple[2]
        for tag_tuple in tag_tuples
        if tag_tuple[1] in allowed_feature_vector_names
    }
    return mlrun.common.schemas.FeatureVectorsTagsOutput(tags=list(tags))


@router.put(
    "/feature-vectors/{name}/references/{reference}",
    response_model=mlrun.common.schemas.FeatureVector,
)
async def store_feature_vector(
    project: str,
    name: str,
    reference: str,
    feature_vector: mlrun.common.schemas.FeatureVector,
    versioned: bool = True,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_vector,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.update,
        auth_info,
    )
    await _verify_feature_vector_features_permissions(
        auth_info, project, feature_vector.dict()
    )
    tag, uid = parse_reference(reference)
    uid = await run_in_threadpool(
        server.api.crud.FeatureStore().store_feature_vector,
        db_session,
        project,
        name,
        feature_vector,
        tag,
        uid,
        versioned,
    )

    return await run_in_threadpool(
        server.api.crud.FeatureStore().get_feature_vector,
        db_session,
        project,
        name,
        tag,
        uid,
    )


@router.patch("/feature-vectors/{name}/references/{reference}")
async def patch_feature_vector(
    project: str,
    name: str,
    feature_vector_patch: dict,
    reference: str,
    patch_mode: mlrun.common.schemas.PatchMode = Header(
        mlrun.common.schemas.PatchMode.replace,
        alias=mlrun.common.schemas.HeaderNames.patch_mode,
    ),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_vector,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.update,
        auth_info,
    )
    await _verify_feature_vector_features_permissions(
        auth_info, project, feature_vector_patch
    )
    tag, uid = parse_reference(reference)
    await run_in_threadpool(
        server.api.crud.FeatureStore().patch_feature_vector,
        db_session,
        project,
        name,
        feature_vector_patch,
        tag,
        uid,
        patch_mode,
    )
    return Response(status_code=HTTPStatus.OK.value)


@router.delete("/feature-vectors/{name}")
@router.delete("/feature-vectors/{name}/references/{reference}")
async def delete_feature_vector(
    project: str,
    name: str,
    reference: str = None,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_vector,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    tag = uid = None
    if reference:
        tag, uid = parse_reference(reference)
    await run_in_threadpool(
        server.api.crud.FeatureStore().delete_feature_vector,
        db_session,
        project,
        name,
        tag,
        uid,
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


async def _verify_feature_vector_features_permissions(
    auth_info: mlrun.common.schemas.AuthInfo, project: str, feature_vector: dict
):
    features = []
    if feature_vector.get("spec", {}).get("features"):
        features.extend(feature_vector["spec"]["features"])
    if feature_vector.get("spec", {}).get("label_feature"):
        features.append(feature_vector["spec"]["label_feature"])
    feature_set_project_to_name_set_map = {}
    for feature in features:
        feature_set_uri, _, _ = mlrun.feature_store.common.parse_feature_string(feature)
        _project, name, _, _ = mlrun.feature_store.common.parse_feature_set_uri(
            feature_set_uri, project
        )
        feature_set_project_to_name_set_map.setdefault(_project, set()).add(name)
    feature_set_project_name_tuples = []
    for _project, names in feature_set_project_to_name_set_map.items():
        for name in names:
            feature_set_project_name_tuples.append((_project, name))
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resources_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.feature_set,
        feature_set_project_name_tuples,
        lambda feature_set_project_name_tuple: (
            feature_set_project_name_tuple[0],
            feature_set_project_name_tuple[1],
        ),
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
