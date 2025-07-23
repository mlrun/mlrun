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

import mlrun.common.constants as mlrun_constants
import mlrun.common.formatters
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.common.schemas.background_task
import mlrun.config
import mlrun.errors
import mlrun.utils
import mlrun.utils.notifications
import mlrun_pipelines.common.models
import mlrun_pipelines.models
import mlrun_pipelines.utils

import framework.api
import framework.api.deps
import framework.api.utils
import framework.utils.auth.verifier
import framework.utils.background_tasks
import framework.utils.notifications
import framework.utils.singletons.k8s
import framework.utils.singletons.project_member
import services.api.crud
from framework.api.utils import log_and_raise

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
    submit_mode: str = fastapi.Query(
        mlrun_constants.WorkflowSubmitMode.rerun, alias="submit-mode"
    ),
    db_session: Session = fastapi.Depends(framework.api.deps.get_db_session),
    client_version: typing.Optional[str] = fastapi.Header(
        None, alias=mlrun.common.schemas.HeaderNames.client_version
    ),
):
    project: mlrun.common.schemas.ProjectOut = (
        await fastapi.concurrency.run_in_threadpool(
            framework.utils.singletons.project_member.get_project_member().get_project,
            db_session=db_session,
            name=project,
            leader_session=auth_info.session,
        )
    )

    # check permission CREATE pipeline
    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            resource_type=mlrun.common.schemas.AuthorizationResourceTypes.pipeline,
            project_name=project.metadata.name,
            resource_name=run_id,
            action=mlrun.common.schemas.AuthorizationAction.create,
            auth_info=auth_info,
        )
    )

    try:
        (
            original_runner,
            original_workflow_id,
        ) = await fastapi.concurrency.run_in_threadpool(
            services.api.crud.Pipelines().get_original_workflow_run,
            db_session=db_session,
            run_id=run_id,
            project=project.metadata.name,
        )
    except mlrun.errors.MLRunNotFoundError:
        original_runner, original_workflow_id = None, None

    # If running in direct mode, or if we couldn’t locate a previous workflow-runner,
    # or if the original runner had no notifications to preserve,
    # skip the RerunRunner orchestration and retry the pipeline directly via the KFP API.
    if (
        submit_mode == mlrun_constants.WorkflowSubmitMode.direct
        or not original_runner
        or not original_runner.spec.notifications
    ):
        mlrun.utils.logger.info("Direct-submitting retry to KFP API", run_id=run_id)
        run_id = await fastapi.concurrency.run_in_threadpool(
            services.api.crud.Pipelines().rerun_pipeline_direct,
            run_id,
            project.metadata.name,
            namespace,
        )

        mlrun.utils.logger.info("Direct retry succeeded", new_pipeline_id=run_id)
        return run_id

    try:
        # Prevent two simultaneous retries for the same original workflow—
        # we lock the original-runner row, mark it retrying, and block any
        # parallel retry requests until it’s cleared.
        await fastapi.concurrency.run_in_threadpool(
            services.api.crud.Pipelines().lock_run_and_mark_retrying,
            db_session=db_session,
            project=project.metadata.name,
            run_id=original_runner.metadata.uid,
        )
    except mlrun.errors.MLRunConflictError as exc:
        try:
            return await fastapi.concurrency.run_in_threadpool(
                services.api.crud.Pipelines().get_running_rerun_runner,
                db_session=db_session,
                project=project.metadata.name,
                original_workflow_id=original_workflow_id,
            )
        except mlrun.errors.MLRunNotFoundError:
            raise mlrun.errors.MLRunConflictError(
                "A retry is already in progress, but no existing rerun was found."
            ) from exc

    try:
        workflow_response: mlrun.common.schemas.WorkflowResponse = (
            await fastapi.concurrency.run_in_threadpool(
                services.api.crud.Pipelines().rerun_pipeline_via_runner,
                db_session=db_session,
                run_id=original_workflow_id,
                project=project,
                original_runner=original_runner,
                auth_info=auth_info,
                client_version=client_version,
            )
        )

        return workflow_response
    except Exception as error:
        mlrun.utils.logger.error(
            "Failed to rerun workflow",
            run_id=original_workflow_id,
            project=project.metadata.name,
            error=mlrun.errors.err_to_str(error),
        )
        log_and_raise(
            reason="Workflow failed",
            error=mlrun.errors.err_to_str(error),
        )


@router.post("/{run_id}/terminate")
async def terminate_pipeline(
    run_id: str,
    project: str,
    background_tasks: BackgroundTasks,
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
            mlrun.common.schemas.AuthorizationAction.delete,
            auth_info,
        )
    )
    run = await fastapi.concurrency.run_in_threadpool(
        func=services.api.crud.pipelines.Pipelines().get_run,
        run_id=run_id,
        project=project,
        namespace=namespace,
    )

    # Check if the pipeline is in a terminable state
    if (
        run.status
        not in mlrun_pipelines.common.models.RunStatuses.terminable_statuses()
    ):
        raise mlrun.errors.MLRunBadRequestError(
            f"Pipeline run {run_id} is not in a terminable state. Current status: {run.status}"
        )

    task = await _terminate_pipeline(
        db_session=db_session,
        background_tasks=background_tasks,
        run_id=run_id,
        project=project,
    )

    return fastapi.Response(
        status_code=202,
        content=task.json(),
        headers={
            "content-type": "application/json",
        },
    )


@router.post(
    "/{run_id}/push-notifications",
    response_model=mlrun.common.schemas.BackgroundTask,
)
async def push_notifications(
    project: str,
    run_id: str,
    background_tasks: BackgroundTasks,
    db_session: sqlalchemy.orm.Session = Depends(framework.api.deps.get_db_session),
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
        None,
        db_session,
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
        services.api.crud.Pipelines().get_formatted_pipeline,
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
        services.api.crud.Pipelines().get_formatted_pipeline,
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
    workflow_manifest = yaml.safe_load(data)
    return services.api.crud.Pipelines().resolve_project_from_workflow_manifest(
        mlrun_pipelines.models.PipelineManifest(workflow_manifest)
    )


def _push_notifications(
    db_session: sqlalchemy.orm.Session,
    run_id: str,
    project: str,
    notifications: typing.Optional[list[mlrun.common.schemas.Notification]] = None,
):
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
        db_session, project, run_id, unmasked_notifications, default_params
    ).push()


async def _terminate_pipeline(
    db_session: sqlalchemy.orm.Session,
    background_tasks: BackgroundTasks,
    run_id: str,
    project: str,
) -> mlrun.common.schemas.BackgroundTask:
    background_task_handler = (
        framework.utils.background_tasks.ProjectBackgroundTasksHandler()
    )
    existing_terminate_pipeline_task = await fastapi.concurrency.run_in_threadpool(
        background_task_handler.get_background_task_by_state_and_labels,
        db_session=db_session,
        status=mlrun.common.schemas.BackgroundTaskState.running,
        labels={
            mlrun.common.schemas.background_task.BackGroundTaskLabel.pipeline: run_id,
        },
    )

    if existing_terminate_pipeline_task is not None:
        mlrun.utils.logger.info(
            "Found existing terminate pipeline task, returning it",
            run_id=run_id,
            task_name=existing_terminate_pipeline_task.metadata.name,
        )
        return existing_terminate_pipeline_task
    else:
        terminate_pipeline_task = await fastapi.concurrency.run_in_threadpool(
            framework.utils.background_tasks.ProjectBackgroundTasksHandler().create_background_task,
            db_session,
            project,
            background_tasks,
            services.api.crud.pipelines.Pipelines().terminate_pipeline,
            mlrun.mlconf.background_tasks.default_timeouts.operations.terminate_pipeline,
            framework.utils.background_tasks.BackgroundTaskKinds.terminate_pipeline.format(
                project,
                run_id,
                time.time(),
            ),
            {
                mlrun.common.schemas.background_task.BackGroundTaskLabel.pipeline: run_id,
            },
            run_id,
            project,
        )

        mlrun.utils.logger.info(
            "Created new terminate pipeline task",
            run_id=run_id,
            task_name=terminate_pipeline_task.metadata.name,
        )
        return terminate_pipeline_task
