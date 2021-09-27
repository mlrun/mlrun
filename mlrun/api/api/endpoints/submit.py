from http import HTTPStatus
from typing import Optional

import fastapi.concurrency
from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.orm import Session

import mlrun.api.api.utils
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.project_member
import mlrun.utils.helpers
from mlrun.api.api import deps
from mlrun.utils import logger

router = APIRouter()


@router.post("/submit")
@router.post("/submit/")
@router.post("/submit_job")
@router.post("/submit_job/")
async def submit_job(
    request: Request,
    username: Optional[str] = Header(None, alias="x-remote-user"),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        mlrun.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value, reason="bad JSON body"
        )

    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        data["task"]["metadata"]["project"],
        auth_info=auth_info,
    )
    function_dict, function_url, task = mlrun.api.api.utils.parse_submit_run_body(data)
    if function_url and "://" not in function_url:
        (
            function_project,
            function_name,
            _,
            _,
        ) = mlrun.utils.helpers.parse_versioned_object_uri(function_url)
        await fastapi.concurrency.run_in_threadpool(
            mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions,
            mlrun.api.schemas.AuthorizationResourceTypes.function,
            function_project,
            function_name,
            mlrun.api.schemas.AuthorizationAction.read,
            auth_info,
        )
    if data.get("schedule"):
        await fastapi.concurrency.run_in_threadpool(
            mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions,
            mlrun.api.schemas.AuthorizationResourceTypes.schedule,
            data["task"]["metadata"]["project"],
            data["task"]["metadata"]["name"],
            mlrun.api.schemas.AuthorizationAction.create,
            auth_info,
        )
    else:
        await fastapi.concurrency.run_in_threadpool(
            mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions,
            mlrun.api.schemas.AuthorizationResourceTypes.run,
            data["task"]["metadata"]["project"],
            "",
            mlrun.api.schemas.AuthorizationAction.create,
            auth_info,
        )

    # enrich job task with the username from the request header
    if username:
        # if task is missing, we don't want to create one
        if "task" in data:
            labels = data["task"].setdefault("metadata", {}).setdefault("labels", {})
            # TODO: remove this duplication
            labels.setdefault("v3io_user", username)
            labels.setdefault("owner", username)

    logger.info("Submit run", data=data)
    response = await mlrun.api.api.utils.submit_run(db_session, auth_info, data)
    return response
