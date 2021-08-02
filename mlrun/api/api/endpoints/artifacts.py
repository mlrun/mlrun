from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.utils.clients.opa
import mlrun.api.utils.singletons.project_member
from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.api.utils.singletons.db import get_db
from mlrun.config import config
from mlrun.utils import logger

router = APIRouter()


# curl -d@/path/to/artifact http://localhost:8080/artifact/p1/7&key=k
@router.post("/artifact/{project}/{uid}/{key:path}")
async def store_artifact(
    request: Request,
    project: str,
    uid: str,
    key: str,
    tag: str = "",
    iter: int = 0,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_verifier.auth_info,
    )
    await run_in_threadpool(
        mlrun.api.utils.clients.opa.Client().query_resource_permissions,
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.api.schemas.AuthorizationAction.store,
        auth_verifier.auth_info,
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


# curl http://localhost:8080/artifact/p1/tags
@router.get("/projects/{project}/artifact-tags")
def list_artifact_tags(
    project: str,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    tag_tuples = get_db().list_artifact_tags(db_session, project)
    artifact_key_to_tag = {tag_tuple[1]: tag_tuple[2] for tag_tuple in tag_tuples}
    allowed_artifact_keys = mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        list(artifact_key_to_tag.keys()),
        lambda artifact_key: (project, artifact_key,),
        auth_verifier.auth_info,
    )
    tags = [
        tag_tuple[2]
        for tag_tuple in tag_tuples
        if tag_tuple[1] in allowed_artifact_keys
    ]
    return {
        "project": project,
        "tags": tags,
    }


# curl http://localhost:8080/projects/my-proj/artifact/key?tag=latest
@router.get("/projects/{project}/artifact/{key:path}")
def get_artifact(
    project: str,
    key: str,
    tag: str = "latest",
    iter: int = 0,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_verifier.auth_info,
    )
    data = mlrun.api.crud.Artifacts().get_artifact(db_session, key, tag, iter, project)
    return {
        "data": data,
    }


# curl -X DELETE http://localhost:8080/artifact/p1&key=k&tag=t
@router.delete("/artifact/{project}/{uid}")
def delete_artifact(
    project: str,
    uid: str,
    key: str,
    tag: str = "",
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        project,
        key,
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_verifier.auth_info,
    )
    mlrun.api.crud.Artifacts().delete_artifact(db_session, key, tag, project)
    return {}


# curl http://localhost:8080/artifacts?project=p1?label=l1
@router.get("/artifacts")
def list_artifacts(
    project: str = config.default_project,
    name: str = None,
    tag: str = None,
    kind: str = None,
    category: schemas.ArtifactCategories = None,
    labels: List[str] = Query([], alias="label"),
    iter: int = Query(None, ge=0),
    best_iteration: bool = Query(False, alias="best-iteration"),
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    artifacts = mlrun.api.crud.Artifacts().list_artifacts(
        db_session,
        project,
        name,
        tag,
        labels,
        kind=kind,
        category=category,
        iter=iter,
        best_iteration=best_iteration,
    )
    artifacts = mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        artifacts,
        lambda artifact: (
            artifact.get("project", mlrun.mlconf.default_project),
            artifact["db_key"],
        ),
        auth_verifier.auth_info,
    )
    return {
        "artifacts": artifacts,
    }


# curl -X DELETE http://localhost:8080/artifacts?project=p1?label=l1
@router.delete("/artifacts")
def delete_artifacts(
    project: str = mlrun.mlconf.default_project,
    name: str = "",
    tag: str = "",
    labels: List[str] = Query([], alias="label"),
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    artifacts = mlrun.api.crud.Artifacts().list_artifacts(
        db_session, project, name, tag, labels
    )
    mlrun.api.utils.clients.opa.Client().query_resources_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.artifact,
        artifacts,
        lambda artifact: (artifact["project"], artifact["db_key"]),
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_verifier.auth_info,
    )
    mlrun.api.crud.Artifacts().delete_artifacts(db_session, project, name, tag, labels)
    return {}
