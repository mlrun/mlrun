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
import datetime
import http
import time
import typing

import fastapi
import fastapi.concurrency
import sqlalchemy.orm
import yaml
from fastapi import BackgroundTasks, Depends
from sqlalchemy.orm import Session

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.utils
import mlrun.utils.notifications
import mlrun_pipelines.models

import framework.api
import framework.api.deps
import framework.api.utils
import framework.utils.auth.verifier
import framework.utils.background_tasks
import framework.utils.notifications
import framework.utils.singletons.k8s
import services.api.crud

router = fastapi.APIRouter(prefix="/projects/{project}/pipelines")


@router.get("", response_model=mlrun.common.schemas.PipelinesOutput)
async def list_pipelines(
    project: str,
    namespace: typing.Optional[str] = None,
    sort_by: str = "",
    page_token: str = "",
    filter_: str = fastapi.Query("", alias="filter"),
    name_contains: str = fastapi.Query("", alias="name-contains"),
    format_: mlrun.common.formatters.PipelineFormat = fastapi.Query(
        mlrun.common.formatters.PipelineFormat.metadata_only, alias="format"
    ),
    page_size: int = fastapi.Query(None, gt=0, le=200),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
):
    if namespace is None:
        namespace = mlrun.config.config.namespace
    allowed_project_names = (
        await services.api.crud.Projects().list_allowed_project_names(
            db_session, auth_info, project=project
        )
    )
    total_size, next_page_token, runs = None, None, []
    if framework.utils.singletons.k8s.get_k8s_helper(
        silent=True
    ).is_running_inside_kubernetes_cluster():
        # we need to resolve the project from the returned run for the opa enforcement (project query param might be
        # "*"), so we can't really get back only the names here
        computed_format = (
            mlrun.common.formatters.PipelineFormat.metadata_only
            if format_ == mlrun.common.formatters.PipelineFormat.name_only
            else format_
        )
        total_size, next_page_token, runs = await fastapi.concurrency.run_in_threadpool(
            services.api.crud.Pipelines().list_pipelines,
            db_session,
            allowed_project_names,
            namespace,
            sort_by,
            page_token,
            filter_,
            name_contains,
            computed_format,
            page_size,
        )
    allowed_runs = await framework.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
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
    request: fastapi.Request,
    experiment_name: str = fastapi.Query("Default", alias="experiment"),
    run_name: str = fastapi.Query("", alias="run"),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
):
    response = await _create_pipeline(
        auth_info=auth_info,
        request=request,
        experiment_name=experiment_name,
        run_name=run_name,
        project=project,
    )
    return response


@router.post("/{run_id}/retry")
async def retry_pipeline(
    run_id: str,
    project: str,
    namespace: str = fastapi.Query(mlrun.config.config.namespace),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
):
    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.pipeline,
            project,
            run_id,
            mlrun.common.schemas.AuthorizationAction.create,
            auth_info,
        )
    )
    run_id = await fastapi.concurrency.run_in_threadpool(
        services.api.crud.Pipelines().retry_pipeline,
        db_session,
        run_id,
        project,
        namespace,
    )
    return run_id


@router.post(
    "/{run_id}/push-notifications",
    response_model=mlrun.common.schemas.BackgroundTask,
)
async def push_notifications(
    project: str,
    run_id: str,
    background_tasks: BackgroundTasks,
    db_session: Session = Depends(framework.api.deps.get_db_session),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    notifications: typing.Optional[list[mlrun.common.schemas.Notification]] = None,
):
    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.pipeline,
            project,
            run_id,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    )

    background_task = await fastapi.concurrency.run_in_threadpool(
        framework.utils.background_tasks.ProjectBackgroundTasksHandler().create_background_task,
        db_session,
        project,
        background_tasks,
        _push_notifications,
        mlrun.mlconf.background_tasks.default_timeouts.push_notifications,
        framework.utils.background_tasks.BackgroundTaskKinds.push_kfp_notification.format(
            project, run_id, time.time()
        ),
        run_id,
        project,
        notifications,
    )
    return background_task


@router.get("/{run_id}")
async def get_pipeline(
    run_id: str,
    project: str,
    namespace: str = fastapi.Query(mlrun.config.config.namespace),
    format_: mlrun.common.formatters.PipelineFormat = fastapi.Query(
        mlrun.common.formatters.PipelineFormat.summary, alias="format"
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
):
    pipeline = await fastapi.concurrency.run_in_threadpool(
        services.api.crud.Pipelines().get_pipeline,
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
        await framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.pipeline,
            project,
            run_id,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    return pipeline


async def _get_pipeline_without_project(
    db_session: sqlalchemy.orm.Session,
    auth_info: mlrun.common.schemas.AuthInfo,
    run_id: str,
    namespace: str,
):
    """
    This function is for when we receive a get pipeline request without the client specifying the project
    So we first get the pipeline, resolve the project out of it, and now that we know the project, we can verify
    permissions
    """
    run = await fastapi.concurrency.run_in_threadpool(
        services.api.crud.Pipelines().get_pipeline,
        db_session,
        run_id,
        namespace=namespace,
        # minimal format that includes the project
        format_=mlrun.common.formatters.PipelineFormat.summary,
    )
    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.pipeline,
            run["run"]["project"],
            run["run"]["id"],
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    )
    return run


async def _create_pipeline(
    auth_info: mlrun.common.schemas.AuthInfo,
    request: fastapi.Request,
    experiment_name: str,
    run_name: str,
    project: typing.Optional[str] = None,
):
    run_name = run_name or experiment_name + " " + datetime.datetime.now().strftime(
        "%Y-%m-%d %H-%M-%S"
    )

    data = await request.body()
    if not data:
        framework.api.utils.log_and_raise(
            http.HTTPStatus.BAD_REQUEST.value, reason="Request body is empty"
        )
    content_type = request.headers.get("content-type", "")

    workflow_project = _try_resolve_project_from_body(content_type, data)
    if project and workflow_project and project != workflow_project:
        framework.api.utils.log_and_raise(
            http.HTTPStatus.BAD_REQUEST.value,
            reason=f"Pipeline contains resources from project {workflow_project} but was requested to be created in "
            f"project {project}",
        )

    project = project or workflow_project
    if not project:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Pipelines can not be created without a project - you are probably running with old client - try upgrade to"
            " the server version"
        )
    else:
        await framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
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

    run = await fastapi.concurrency.run_in_threadpool(
        services.api.crud.Pipelines().create_pipeline,
        experiment_name,
        run_name,
        content_type,
        data,
        arguments,
    )

    return {
        "id": run.id,
        "name": run.name,
    }


def _try_resolve_project_from_body(
    content_type: str, data: bytes
) -> typing.Optional[str]:
    if "/yaml" not in content_type:
        mlrun.utils.logger.warning(
            "Could not resolve project from body, unsupported content type",
            content_type=content_type,
        )
        return None
    workflow_manifest = yaml.load(data, Loader=yaml.FullLoader)
    return services.api.crud.Pipelines().resolve_project_from_workflow_manifest(
        mlrun_pipelines.models.PipelineManifest(workflow_manifest)
    )


def _push_notifications(run_id, project, notifications):
    if not notifications:
        return
    unmasked_notifications = []
    for notification in notifications:
        try:
            unmasked_notifications.append(
                framework.utils.notifications.unmask_notification_params_secret(
                    project, notification
                )
            )
        except Exception as exc:
            mlrun.utils.logger.warning(
                "Failed to unmask notification params secret",
                notification=notification,
                exc=exc,
            )
    run_notification_pusher = (
        framework.utils.notifications.notification_pusher.RunNotificationPusher
    )
    default_params = run_notification_pusher.resolve_notifications_default_params()
    framework.utils.notifications.notification_pusher.KFPNotificationPusher(
        project, run_id, unmasked_notifications, default_params
    ).push()
