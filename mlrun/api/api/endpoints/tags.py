import typing

import fastapi
import fastapi.concurrency
import sqlalchemy.orm
from fastapi import APIRouter, Depends, Query, Request

import mlrun.api.api.deps
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.project_member

router = fastapi.APIRouter()


@router.post("/projects/{project}/tags/{tag}")
async def store_tag(
    project: str,
    tag: str,
    objects: mlrun.api.schemas.TagsObjects,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    for object_list in objects.objects:
        # check permission per object type
        await fastapi.concurrency.run_in_threadpool(
            mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions,
            object_list.kind,
            project,
            object_list.kind,  # we don't really check per object, just the kind, so passing kind just as a place holder
            mlrun.api.schemas.AuthorizationAction.store,
            auth_info,
        )
