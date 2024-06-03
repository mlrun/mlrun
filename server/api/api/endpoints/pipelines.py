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
import ast
import typing
from datetime import datetime
from http import HTTPStatus

import yaml
from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from mlrun_pipelines.models import PipelineManifest
from sqlalchemy.orm import Session

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.errors
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.singletons.k8s
from mlrun.config import config
from mlrun.utils import logger
from server.api.api import deps
from server.api.api.utils import log_and_raise

router = APIRouter(prefix="/projects/{project}/pipelines")


@router.get("", response_model=mlrun.common.schemas.PipelinesOutput)
async def list_pipelines(
    project: str,
    namespace: str = None,
    sort_by: str = "",
    page_token: str = "",
    filter_: str = Query("", alias="filter"),
    name_contains: str = Query("", alias="name-contains"),
    format_: mlrun.common.formatters.PipelineFormat = Query(
        mlrun.common.formatters.PipelineFormat.metadata_only, alias="format"
    ),
    page_size: int = Query(None, gt=0, le=200),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    if namespace is None:
        namespace = config.namespace
    if project != "*":
        await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
            project,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    total_size, next_page_token, runs = None, None, []
    if server.api.utils.singletons.k8s.get_k8s_helper(
        silent=True
    ).is_running_inside_kubernetes_cluster():
        # we need to resolve the project from the returned run for the opa enforcement (project query param might be
        # "*"), so we can't really get back only the names here
        computed_format = (
            mlrun.common.formatters.PipelineFormat.metadata_only
            if format_ == mlrun.common.formatters.PipelineFormat.name_only
            else format_
        )
        total_size, next_page_token, runs = await run_in_threadpool(
            server.api.crud.Pipelines().list_pipelines,
            db_session,
            project,
            namespace,
            sort_by,
            page_token,
            filter_,
            name_contains,
            computed_format,
            page_size,
        )
    allowed_runs = await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.pipeline,
        runs,
        lambda run: (
            run["project"],
            run["id"],
        ),
        auth_info,
    )
    if format_ == mlrun.common.formatters.PipelineFormat.name_only:
        allowed_runs = [
            mlrun.common.formatters.PipelineFormat.format_obj(run, format_)
            for run in allowed_runs
        ]
    return mlrun.common.schemas.PipelinesOutput(
        runs=allowed_runs,
        total_size=total_size or 0,
        next_page_token=next_page_token or None,
    )


@router.post("")
async def create_pipeline(
    project: str,
    request: Request,
    namespace: str = None,
    experiment_name: str = Query("Default", alias="experiment"),
    run_name: str = Query("", alias="run"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    if namespace is None:
        namespace = config.namespace
    response = await _create_pipeline(
        auth_info, request, namespace, experiment_name, run_name, project
    )
    return response


@router.get("/{run_id}")
async def get_pipeline(
    run_id: str,
    project: str,
    namespace: str = Query(config.namespace),
    format_: mlrun.common.formatters.PipelineFormat = Query(
        mlrun.common.formatters.PipelineFormat.summary, alias="format"
    ),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    pipeline = await run_in_threadpool(
        server.api.crud.Pipelines().get_pipeline,
        db_session,
        run_id,
        project,
        namespace,
        format_,
    )
    if project == "*":
        # In some flows the user may use SDK functions that won't require them to specify the pipeline's project (for
        # backwards compatibility reasons), so the client will just send * in the project, in that case we use the
        # legacy flow in which we first get the pipeline, resolve the project out of it, and only then query permissions
        # we don't use the return value from this function since the user may have asked for a different format than
        # summary which is the one used inside
        await _get_pipeline_without_project(db_session, auth_info, run_id, namespace)
    else:
        await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.pipeline,
            project,
            run_id,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    return pipeline


async def _get_pipeline_without_project(
    db_session: Session,
    auth_info: mlrun.common.schemas.AuthInfo,
    run_id: str,
    namespace: str,
):
    """
    This function is for when we receive a get pipeline request without the client specifying the project
    So we first get the pipeline, resolve the project out of it, and now that we know the project, we can verify
    permissions
    """
    run = await run_in_threadpool(
        server.api.crud.Pipelines().get_pipeline,
        db_session,
        run_id,
        namespace=namespace,
        # minimal format that includes the project
        format_=mlrun.common.formatters.PipelineFormat.summary,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.pipeline,
        run["run"]["project"],
        run["run"]["id"],
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return run


async def _create_pipeline(
    auth_info: mlrun.common.schemas.AuthInfo,
    request: Request,
    namespace: str,
    experiment_name: str,
    run_name: str,
    project: typing.Optional[str] = None,
):
    run_name = run_name or experiment_name + " " + datetime.now().strftime(
        "%Y-%m-%d %H-%M-%S"
    )

    data = await request.body()
    if not data:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="Request body is empty")
    content_type = request.headers.get("content-type", "")

    workflow_project = _try_resolve_project_from_body(content_type, data)
    if project and project != workflow_project:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="Some resources in the workflow are from a different project than the one specified in the URL. "
                   "Cross project pipelines are not supported.",
        )

    project = project or workflow_project
    if not project:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Pipelines can not be created without a project - you are probably running with old client - try upgrade to"
            " the server version"
        )
    else:
        await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.pipeline,
            project,
            "",
            mlrun.common.schemas.AuthorizationAction.create,
            auth_info,
        )

    arguments = {}
    arguments_data = request.headers.get(
        mlrun.common.schemas.HeaderNames.pipeline_arguments
    )
    if arguments_data:
        arguments = ast.literal_eval(arguments_data)

    run = await run_in_threadpool(
        server.api.crud.Pipelines().create_pipeline,
        experiment_name,
        run_name,
        content_type,
        data,
        arguments,
        namespace,
    )

    return {
        "id": run.id,
        "name": run.name,
    }


def _try_resolve_project_from_body(
    content_type: str, data: bytes
) -> typing.Optional[str]:
    if "/yaml" not in content_type:
        logger.warning(
            "Could not resolve project from body, unsupported content type",
            content_type=content_type,
        )
        return None
    workflow_manifest = yaml.load(data, Loader=yaml.FullLoader)
    return server.api.crud.Pipelines().resolve_project_from_workflow_manifest(
        PipelineManifest(workflow_manifest)
    )
