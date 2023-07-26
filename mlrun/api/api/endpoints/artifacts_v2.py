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

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.project_member
import mlrun.common.schemas
from mlrun.api.api import deps
from mlrun.api.api.utils import (
    artifact_project_and_resource_name_extractor,
    log_and_raise,
)
from mlrun.common.schemas.artifact import ArtifactsFormat
from mlrun.config import config
from mlrun.utils import logger

router = APIRouter()


@router.post("/projects/{project}/artifacts")
async def create_artifact(
    request: Request,
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    key = data.get("metadata").get("key", None)
    tag = data.get("metadata").get("tag", None)
    iteration = data.get("metadata").get("iter", None)
    logger.debug("Storing artifact", project=project, key=key, tag=tag, iter=iter)
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    await run_in_threadpool(
        mlrun.api.crud.Artifacts().store_artifact,
        db_session,
        key,
        data,
        None,  # uid is auto generated
        tag,
        iteration,
        project,
    )
    return {}


@router.put("/projects/{project}/artifacts/{key:path}")
async def store_artifact(
    request: Request,
    project: str,
    key: str,
    tree: str = None,
    tag: str = None,
    iter: int = 0,
    uid: str = None,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.debug("Updating artifact", project=project, key=key, tag=tag, iter=iter)
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    await run_in_threadpool(
        mlrun.api.crud.Artifacts().store_artifact,
        db_session,
        key,
        data,
        uid,
        tag,
        iter,
        project,
        tree,
    )
    return {}


@router.get("/projects/{project}/artifacts")
async def list_artifacts(
    project: str = None,
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
    if project is None:
        project = config.default_project
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
        tree=tree,
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
    iter: int = 0,
    uid: str = None,
    format_: ArtifactsFormat = Query(ArtifactsFormat.full, alias="format"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    data = await run_in_threadpool(
        mlrun.api.crud.Artifacts().get_artifact,
        db_session,
        key,
        tag,
        iter,
        project,
        format_,
        tree,
        uid,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return {
        "data": data,
    }


@router.delete("/projects/{project}/artifacts/{key:path}")
async def delete_artifact(
    project: str,
    key: str,
    tree: str = None,
    tag: str = None,
    uid: str = None,
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
        uid,
        tree,
    )
    return {}


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
        tree=tree,
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
    )
    return {}
