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

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.formatters
import mlrun.common.schemas
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.singletons.project_member
from mlrun.config import config
from mlrun.utils import logger
from server.api.api import deps
from server.api.api.utils import (
    artifact_project_and_resource_name_extractor,
    log_and_raise,
)

router = APIRouter()


@router.post("/projects/{project}/artifacts/{uid}/{key:path}")
async def store_artifact(
    request: Request,
    project: str,
    uid: str,
    key: str,
    tag: str = "",
    iter: int = 0,
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
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )

    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    # the v1 artifacts `uid` parameter is essentially the `tree` parameter in v2
    tree = uid

    logger.debug(
        "Storing artifact", project=project, tree=tree, key=key, tag=tag, iter=iter
    )
    await run_in_threadpool(
        server.api.crud.Artifacts().store_artifact,
        db_session,
        key,
        data,
        tag=tag,
        iter=iter,
        project=project,
        producer_id=tree,
        auth_info=auth_info,
    )
    return {}


@router.get("/projects/{project}/artifact-tags")
async def list_artifact_tags(
    project: str,
    category: mlrun.common.schemas.ArtifactCategories = None,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    # verify that the user has permissions to read the project's artifacts
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        "",
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    tags = await run_in_threadpool(
        server.api.crud.Artifacts().list_artifact_tags, db_session, project, category
    )

    return {
        "project": project,
        "tags": tags,
    }


@router.get("/projects/{project}/artifacts/{key:path}")
async def get_artifact(
    project: str,
    key: str,
    tag: str = "latest",
    iter: int = 0,
    format_: str = Query(mlrun.common.formatters.ArtifactFormat.full, alias="format"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    try:
        data = await run_in_threadpool(
            server.api.crud.Artifacts().get_artifact,
            db_session,
            key,
            tag=tag,
            iter=iter,
            project=project,
            format_=format_,
        )
    except mlrun.errors.MLRunNotFoundError:
        logger.debug(
            "Artifact not found, trying to get it with producer_id=tag to support older versions",
            project=project,
            key=key,
            tag=tag,
        )

        # in earlier versions, producer_id and tag got confused with each other,
        # so we try to get the artifact with the given tag as the producer_id before returning an empty response
        data = await run_in_threadpool(
            server.api.crud.Artifacts().get_artifact,
            db_session,
            key,
            iter=iter,
            project=project,
            format_=format_,
            producer_id=tag,
        )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return {
        "data": data,
    }


@router.delete("/projects/{project}/artifacts/{uid}")
async def delete_artifact(
    project: str,
    uid: str,
    key: str,
    tag: str = "",
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    await run_in_threadpool(
        server.api.crud.Artifacts().delete_artifact, db_session, key, tag, project
    )
    return {}


@router.get("/projects/{project}/artifacts")
async def list_artifacts(
    project: str = None,
    name: str = None,
    tag: str = None,
    kind: str = None,
    category: mlrun.common.schemas.ArtifactCategories = None,
    labels: list[str] = Query([], alias="label"),
    iter: int = Query(None, ge=0),
    best_iteration: bool = Query(False, alias="best-iteration"),
    format_: str = Query(mlrun.common.formatters.ArtifactFormat.full, alias="format"),
    limit: int = Query(None),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    if project is None:
        project = config.default_project
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
        kind=kind,
        category=category,
        iter=iter,
        best_iteration=best_iteration,
        format_=format_,
        limit=limit,
    )

    if not artifacts and tag:
        # in earlier versions, producer_id and tag got confused with each other,
        # so we search for results with the given tag as the producer_id
        artifacts = await run_in_threadpool(
            server.api.crud.Artifacts().list_artifacts,
            db_session,
            project,
            name,
            "",
            labels,
            kind=kind,
            category=category,
            iter=iter,
            best_iteration=best_iteration,
            format_=format_,
            producer_id=tag,
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


@router.delete("/projects/{project}/artifacts")
async def delete_artifacts(
    project: str = None,
    name: str = "",
    tag: str = "",
    labels: list[str] = Query([], alias="label"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    return await _delete_artifacts(
        project=project or mlrun.mlconf.default_project,
        name=name,
        tag=tag,
        labels=labels,
        auth_info=auth_info,
        db_session=db_session,
    )


async def _delete_artifacts(
    project: str = None,
    name: str = None,
    tag: str = None,
    labels: list[str] = None,
    auth_info: mlrun.common.schemas.AuthInfo = None,
    db_session: Session = None,
):
    artifacts = await run_in_threadpool(
        server.api.crud.Artifacts().list_artifacts,
        db_session,
        project,
        name,
        tag,
        labels,
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
    )
    return {}
