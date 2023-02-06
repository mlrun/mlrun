# Copyright 2018 Iguazio
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
from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.api.schemas.artifact import ArtifactsFormat
from mlrun.config import config
from mlrun.utils import is_legacy_artifact, logger

router = APIRouter()


# TODO /artifact/{project}/{uid}/{key:path} should be deprecated in 1.4
@router.post("/artifact/{project}/{uid}/{key:path}")
@router.post("/projects/{project}/artifacts/{uid}/{key:path}")
async def store_artifact(
    request: Request,
    project: str,
    uid: str,
    key: str,
    tag: str = "",
    iter: int = 0,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.api.schemas.AuthorizationAction.store,
        auth_info,
    )

    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.debug("Storing artifact", data=data)
    await run_in_threadpool(
        mlrun.api.crud.Artifacts().store_artifact,
        db_session,
        key,
        data,
        uid,
        tag,
        iter,
        project,
    )
    return {}


@router.get("/projects/{project}/artifact-tags")
async def list_artifact_tags(
    project: str,
    category: schemas.ArtifactCategories = None,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    tag_tuples = await run_in_threadpool(
        mlrun.api.crud.Artifacts().list_artifact_tags, db_session, project, category
    )
    artifact_key_to_tag = {tag_tuple[1]: tag_tuple[2] for tag_tuple in tag_tuples}
    allowed_artifact_keys = await mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        list(artifact_key_to_tag.keys()),
        lambda artifact_key: (
            project,
            artifact_key,
        ),
        auth_info,
    )
    tags = [
        tag_tuple[2]
        for tag_tuple in tag_tuples
        if tag_tuple[1] in allowed_artifact_keys
    ]
    return {
        "project": project,
        # Remove duplicities
        "tags": list(set(tags)),
    }


# TODO /projects/{project}/artifact/{key:path} should be deprecated in 1.4
@router.get("/projects/{project}/artifact/{key:path}")
@router.get("/projects/{project}/artifacts/{key:path}")
async def get_artifact(
    project: str,
    key: str,
    tag: str = "latest",
    iter: int = 0,
    format_: ArtifactsFormat = Query(ArtifactsFormat.full, alias="format"),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
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
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    return {
        "data": data,
    }


# TODO /artifact/{project}/{uid} should be deprecated in 1.4
@router.delete("/artifact/{project}/{uid}")
@router.delete("/projects/{project}/artifacts/{uid}")
async def delete_artifact(
    project: str,
    uid: str,
    key: str,
    tag: str = "",
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_info,
    )
    await run_in_threadpool(
        mlrun.api.crud.Artifacts().delete_artifact, db_session, key, tag, project
    )
    return {}


# TODO /artifacts should be deprecated in 1.4
@router.get("/artifacts")
@router.get("/projects/{project}/artifacts")
async def list_artifacts(
    project: str = None,
    name: str = None,
    tag: str = None,
    kind: str = None,
    category: schemas.ArtifactCategories = None,
    labels: List[str] = Query([], alias="label"),
    iter: int = Query(None, ge=0),
    best_iteration: bool = Query(False, alias="best-iteration"),
    format_: ArtifactsFormat = Query(ArtifactsFormat.full, alias="format"),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    if project is None:
        project = config.default_project
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.api.schemas.AuthorizationAction.read,
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
    )

    artifacts = await mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        artifacts,
        _artifact_project_and_resource_name_extractor,
        auth_info,
    )
    return {
        "artifacts": artifacts,
    }


# TODO /artifacts should be deprecated in 1.4
@router.delete("/artifacts")
async def delete_artifacts_legacy(
    project: str = mlrun.mlconf.default_project,
    name: str = "",
    tag: str = "",
    labels: List[str] = Query([], alias="label"),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    return await _delete_artifacts(
        project=project,
        name=name,
        tag=tag,
        labels=labels,
        auth_info=auth_info,
        db_session=db_session,
    )


@router.delete("/projects/{project}/artifacts")
async def delete_artifacts(
    project: str = mlrun.mlconf.default_project,
    name: str = "",
    tag: str = "",
    labels: List[str] = Query([], alias="label"),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    return await _delete_artifacts(
        project=project,
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
    labels: List[str] = None,
    auth_info: mlrun.api.schemas.AuthInfo = None,
    db_session: Session = None,
):
    artifacts = await run_in_threadpool(
        mlrun.api.crud.Artifacts().list_artifacts,
        db_session,
        project,
        name,
        tag,
        labels,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resources_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        artifacts,
        _artifact_project_and_resource_name_extractor,
        mlrun.api.schemas.AuthorizationAction.delete,
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


# Extract project and resource name from legacy artifact structure as well as from new format
def _artifact_project_and_resource_name_extractor(artifact):
    if is_legacy_artifact(artifact):
        return artifact.get("project", mlrun.mlconf.default_project), artifact["db_key"]
    else:
        return (
            artifact.get("metadata").get("project", mlrun.mlconf.default_project),
            artifact.get("spec")["db_key"],
        )
