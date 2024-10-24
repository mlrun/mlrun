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
from http import HTTPStatus

from fastapi import APIRouter, Depends, Query, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.formatters
import mlrun.common.schemas
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.singletons.project_member
from mlrun.common.schemas.artifact import ArtifactsDeletionStrategies
from mlrun.utils import logger
from server.api.api import deps
from server.api.api.utils import artifact_project_and_resource_name_extractor

router = APIRouter()


@router.post("/projects/{project}/artifacts", status_code=HTTPStatus.CREATED.value)
async def create_artifact(
    project: str,
    artifact: mlrun.common.schemas.Artifact,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    key = artifact.metadata.key or None
    tag = artifact.metadata.tag or None
    iteration = artifact.metadata.iter or 0
    tree = artifact.metadata.tree or None
    logger.debug("Creating artifact", project=project, key=key, tag=tag, iter=iteration)
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    artifact_uid = await run_in_threadpool(
        server.api.crud.Artifacts().create_artifact,
        db_session,
        key,
        artifact.dict(exclude_none=True),
        tag,
        iteration,
        producer_id=tree,
        project=project,
        auth_info=auth_info,
    )
    return await run_in_threadpool(
        server.api.crud.Artifacts().get_artifact,
        db_session,
        key,
        tag,
        iteration,
        project,
        producer_id=tree,
        object_uid=artifact_uid,
    )


@router.put("/projects/{project}/artifacts/{key:path}")
async def store_artifact(
    project: str,
    artifact: mlrun.common.schemas.Artifact,
    key: str,
    tree: str = None,
    tag: str = None,
    iter: int = None,
    object_uid: str = Query(None, alias="object-uid"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    producer_id = tree
    if iter is None:
        iter = artifact.metadata.iter or 0
    logger.debug(
        "Storing artifact",
        project=project,
        key=key,
        tag=tag,
        producer_id=producer_id,
        iter=iter,
    )

    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    artifact_uid = await run_in_threadpool(
        server.api.crud.Artifacts().store_artifact,
        db_session,
        key,
        artifact.dict(exclude_none=True),
        object_uid,
        tag,
        iter,
        project,
        producer_id=producer_id,
        auth_info=auth_info,
    )
    return await run_in_threadpool(
        server.api.crud.Artifacts().get_artifact,
        db_session,
        key,
        tag,
        iter,
        project,
        producer_id=producer_id,
        object_uid=artifact_uid,
    )


@router.get("/projects/{project}/artifacts")
async def list_artifacts(
    project: str,
    name: str = None,
    tag: str = None,
    kind: str = None,
    category: mlrun.common.schemas.ArtifactCategories = None,
    labels: list[str] = Query([], alias="label"),
    iter: int = Query(None, ge=0),
    tree: str = None,
    producer_uri: str = None,
    best_iteration: bool = Query(False, alias="best-iteration"),
    format_: str = Query(mlrun.common.formatters.ArtifactFormat.full, alias="format"),
    limit: int = Query(None),
    since: str = None,
    until: str = None,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    artifacts = await run_in_threadpool(
        server.api.crud.Artifacts().list_artifacts,
        db_session,
        project,
        name,
        tag,
        labels,
        since=mlrun.utils.datetime_from_iso(since),
        until=mlrun.utils.datetime_from_iso(until),
        kind=kind,
        category=category,
        iter=iter,
        best_iteration=best_iteration,
        format_=format_,
        producer_id=tree,
        producer_uri=producer_uri,
        limit=limit,
    )

    artifacts = await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        artifacts,
        artifact_project_and_resource_name_extractor,
        auth_info,
    )
    return {
        "artifacts": artifacts,
    }


@router.get("/projects/{project}/artifacts/{key:path}")
async def get_artifact(
    project: str,
    key: str,
    tree: str = None,
    tag: str = None,
    iter: int = None,
    object_uid: str = Query(None, alias="object-uid"),
    # TODO: remove deprecated uid parameter in 1.9.0
    # we support both uid and object-uid for backward compatibility
    uid: str = Query(
        None,
        deprecated=True,
        description="Use object-uid instead, will be removed in the 1.9.0",
    ),
    format_: str = Query(mlrun.common.formatters.ArtifactFormat.full, alias="format"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    artifact = await run_in_threadpool(
        server.api.crud.Artifacts().get_artifact,
        db_session,
        key,
        tag,
        iter,
        project,
        format_,
        producer_id=tree,
        object_uid=object_uid or uid,
    )
    return artifact


@router.delete("/projects/{project}/artifacts/{key:path}")
async def delete_artifact(
    project: str,
    key: str,
    tree: str = None,
    tag: str = None,
    object_uid: str = Query(None, alias="object-uid"),
    # TODO: remove deprecated uid parameter in 1.9.0
    # we support both uid and object-uid for backward compatibility
    uid: str = Query(
        None,
        deprecated=True,
        description="Use object-uid instead, will be removed in the 1.9.0",
    ),
    iteration: int = Query(None, alias="iter"),
    deletion_strategy: ArtifactsDeletionStrategies = ArtifactsDeletionStrategies.metadata_only,
    secrets: dict = None,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    logger.debug(
        "Deleting artifact",
        project=project,
        key=key,
        tag=tag,
        producer_id=tree,
        deletion_strategy=deletion_strategy,
        iteration=iteration,
        object_uid=object_uid or uid,
    )

    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    await run_in_threadpool(
        server.api.crud.Artifacts().delete_artifact,
        db_session=db_session,
        key=key,
        tag=tag,
        project=project,
        object_uid=object_uid or uid,
        producer_id=tree,
        deletion_strategy=deletion_strategy,
        secrets=secrets,
        auth_info=auth_info,
        iteration=iteration,
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.delete("/projects/{project}/artifacts")
async def delete_artifacts(
    project: str = None,
    name: str = "",
    tag: str = "",
    tree: str = None,
    labels: list[str] = Query([], alias="label"),
    limit: int = Query(None),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    project = project or mlrun.mlconf.default_project
    artifacts = await run_in_threadpool(
        server.api.crud.Artifacts().list_artifacts,
        db_session,
        project,
        name,
        tag,
        labels,
        producer_id=tree,
        limit=limit,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resources_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        artifacts,
        artifact_project_and_resource_name_extractor,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    await run_in_threadpool(
        server.api.crud.Artifacts().delete_artifacts,
        db_session,
        project,
        name,
        tag,
        labels,
        auth_info,
        producer_id=tree,
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)
