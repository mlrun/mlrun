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
from typing import List

from fastapi import APIRouter, Depends, Query, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.project_member
import mlrun.common.schemas
from mlrun.api.api import deps
from mlrun.api.api.utils import artifact_project_and_resource_name_extractor
from mlrun.common.schemas.artifact import ArtifactsFormat
from mlrun.utils import logger

router = APIRouter()


@router.post("/projects/{project}/artifacts")
async def create_artifact(
    project: str,
    artifact: mlrun.common.schemas.Artifact,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    key = artifact.metadata.key or None
    tag = artifact.metadata.tag or None
    iteration = artifact.metadata.iter or 0
    tree = artifact.metadata.tree or None
    logger.debug("Creating artifact", project=project, key=key, tag=tag, iter=iteration)
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    artifact_uid = await run_in_threadpool(
        mlrun.api.crud.Artifacts().create_artifact,
        db_session,
        key,
        artifact.dict(exclude_none=True),
        tag,
        iteration,
        tree,
        project,
    )
    return await run_in_threadpool(
        mlrun.api.crud.Artifacts().get_artifact,
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
    iter: int = 0,
    object_uid: str = None,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    logger.debug("Storing artifact", project=project, key=key, tag=tag, iter=iter)
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    producer_id = tree
    artifact_uid = await run_in_threadpool(
        mlrun.api.crud.Artifacts().store_artifact,
        db_session,
        key,
        artifact.dict(exclude_none=True),
        object_uid,
        tag,
        iter,
        project,
        producer_id=producer_id,
    )
    return await run_in_threadpool(
        mlrun.api.crud.Artifacts().get_artifact,
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
    labels: List[str] = Query([], alias="label"),
    iter: int = Query(None, ge=0),
    tree: str = None,
    best_iteration: bool = Query(False, alias="best-iteration"),
    format_: ArtifactsFormat = Query(ArtifactsFormat.full, alias="format"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    artifacts = await run_in_threadpool(
        mlrun.api.crud.Artifacts().list_artifacts,
        db_session,
        project,
        name,
        tag,
        labels,
        kind=kind,
        category=category,
        iter=iter,
        best_iteration=best_iteration,
        format_=format_,
        producer_id=tree,
    )

    artifacts = await mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
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
    object_uid: str = None,
    format_: ArtifactsFormat = Query(ArtifactsFormat.full, alias="format"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    artifact = await run_in_threadpool(
        mlrun.api.crud.Artifacts().get_artifact,
        db_session,
        key,
        tag,
        iter,
        project,
        format_,
        producer_id=tree,
        object_uid=object_uid,
    )
    return artifact


@router.delete("/projects/{project}/artifacts/{key:path}")
async def delete_artifact(
    project: str,
    key: str,
    tree: str = None,
    tag: str = None,
    object_uid: str = None,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    await run_in_threadpool(
        mlrun.api.crud.Artifacts().delete_artifact,
        db_session,
        key,
        tag,
        project,
        object_uid,
        producer_id=tree,
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.delete("/projects/{project}/artifacts")
async def delete_artifacts(
    project: str = mlrun.mlconf.default_project,
    name: str = "",
    tag: str = "",
    tree: str = None,
    labels: List[str] = Query([], alias="label"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    artifacts = await run_in_threadpool(
        mlrun.api.crud.Artifacts().list_artifacts,
        db_session,
        project,
        name,
        tag,
        labels,
        producer_id=tree,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resources_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        artifacts,
        artifact_project_and_resource_name_extractor,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    await run_in_threadpool(
        mlrun.api.crud.Artifacts().delete_artifacts,
        db_session,
        project,
        name,
        tag,
        labels,
        auth_info,
        producer_id=tree,
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)
