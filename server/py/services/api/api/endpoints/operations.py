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
import http
import typing

import fastapi
from fastapi import Depends
from fastapi.concurrency import run_in_threadpool
from kfp_server_api import ApiException

import mlrun.common.schemas
from mlrun.utils import logger

import framework.api.deps
import framework.utils.background_tasks
import framework.utils.clients.chief
import framework.utils.clients.iguazio as iguazio_client
import framework.utils.notifications.notification_pusher as notification_pusher
import framework.utils.singletons
import services.api.initial_data

router = fastapi.APIRouter(prefix="/operations")


current_migration_background_task_name = None
current_refresh_smtp_configuration_task_name = None


@router.post(
    "/migrations",
    responses={
        http.HTTPStatus.OK.value: {},
        http.HTTPStatus.ACCEPTED.value: {"model": mlrun.common.schemas.BackgroundTask},
    },
)
async def trigger_migrations(
    background_tasks: fastapi.BackgroundTasks,
    response: fastapi.Response,
    request: fastapi.Request,
):
    # only chief can execute migrations, redirecting request to chief
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info("Requesting to trigger migrations, re-routing to chief")
        chief_client = framework.utils.clients.chief.Client()
        return await chief_client.trigger_migrations(request)

    # we didn't yet decide who should have permissions to such actions, therefore no authorization at the moment
    # note in api.py we do declare to use the authenticate_request dependency - meaning we do have authentication
    global current_migration_background_task_name

    task_callback, background_task, task_name = await run_in_threadpool(
        _get_or_create_migration_background_task,
        current_migration_background_task_name,
    )
    if not task_callback and not background_task:
        # Not waiting for migrations, returning OK
        return fastapi.Response(status_code=http.HTTPStatus.OK.value)

    if not background_task:
        # No task in progress, creating a new one
        background_tasks.add_task(task_callback)
        background_task = framework.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task(
            task_name
        )

    response.status_code = http.HTTPStatus.ACCEPTED.value
    current_migration_background_task_name = background_task.metadata.name
    return background_task


@router.post(
    "/refresh-smtp-configuration",
    responses={
        http.HTTPStatus.OK.value: {},
        http.HTTPStatus.ACCEPTED.value: {"model": mlrun.common.schemas.BackgroundTask},
    },
)
async def refresh_smtp_configuration(
    background_tasks: fastapi.BackgroundTasks,
    response: fastapi.Response,
    request: fastapi.Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(
        framework.api.deps.authenticate_request
    ),
):
    # we want that only the chief will store the secret in the k8s secret store for preventing
    # race conditions
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info("Requesting to refresh SMTP configuration, re-routing to chief")
        chief_client = framework.utils.clients.chief.Client()
        return await chief_client.refresh_smtp_configuration(request)

    if not framework.utils.singletons.k8s.get_k8s_helper().is_running_inside_kubernetes_cluster():
        raise mlrun.errors.MLRunPreconditionFailedError(
            "SMTP configuration can be refreshed only when running inside a Kubernetes cluster"
        )

    task_callback, task_name = await run_in_threadpool(
        _create_refresh_smtp_configuration_background_task,
        auth_info.session,
    )

    background_tasks.add_task(task_callback)
    background_task = framework.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task(
        task_name
    )

    response.status_code = http.HTTPStatus.ACCEPTED.value
    return background_task


def _get_or_create_migration_background_task(
    task_name: str,
) -> tuple[
    typing.Optional[typing.Callable],
    typing.Optional[mlrun.common.schemas.BackgroundTask],
    str,
]:
    if (
        mlrun.mlconf.httpdb.state
        == mlrun.common.schemas.APIStates.migrations_in_progress
    ):
        background_task = framework.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task(
            task_name
        )
        return None, background_task, task_name
    elif mlrun.mlconf.httpdb.state == mlrun.common.schemas.APIStates.migrations_failed:
        raise mlrun.errors.MLRunPreconditionFailedError(
            "Migrations were already triggered and failed. Restart the API to retry"
        )
    elif (
        mlrun.mlconf.httpdb.state
        != mlrun.common.schemas.APIStates.waiting_for_migrations
    ):
        return None, None, ""

    logger.info("Starting the migration process")
    (
        task,
        task_name,
    ) = framework.utils.background_tasks.InternalBackgroundTasksHandler().create_background_task(
        framework.utils.background_tasks.BackgroundTaskKinds.db_migrations,
        None,
        _perform_migration,
    )
    return task, None, task_name


async def _perform_migration():
    # import here to prevent import cycle
    from services.api.daemon import daemon

    await run_in_threadpool(
        services.api.initial_data.init_data, perform_migrations_if_needed=True
    )
    await daemon.service.move_service_to_online()
    mlrun.mlconf.httpdb.state = mlrun.common.schemas.APIStates.online


def _create_refresh_smtp_configuration_background_task(
    session: str,
) -> tuple[
    typing.Optional[typing.Callable],
    str,
]:
    logger.info("Starting the SMTP configuration refresh process")
    (
        task,
        task_name,
    ) = framework.utils.background_tasks.InternalBackgroundTasksHandler().create_background_task(
        framework.utils.background_tasks.BackgroundTaskKinds.refresh_smtp_configuration,
        None,
        _perform_refresh_smtp,
        session=session,
    )
    return task, task_name


async def _perform_refresh_smtp(session: str):
    iguazio_client_instance = iguazio_client.Client()
    try:
        returned_smtp_configuration = iguazio_client_instance.get_smtp_configuration(
            session
        )
    except mlrun.errors.MLRunInternalServerError as exc:
        logger.warning(
            "Failed to get SMTP configuration from Iguazio",
            exc=mlrun.errors.err_to_str(exc),
        )
        raise

    updated_params = {
        "server_host": returned_smtp_configuration.host,
        "server_port": str(returned_smtp_configuration.port),
        "sender_address": returned_smtp_configuration.sender_address,
        "username": returned_smtp_configuration.auth_username,
        "password": returned_smtp_configuration.auth_password,
    }
    _store_mail_notifications_default_params_to_secret(updated_params)
    notification_pusher.RunNotificationPusher.get_mail_notification_default_params(
        refresh=True
    )


def _store_mail_notifications_default_params_to_secret(default_params: dict):
    smtp_config_secret_name = mlrun.mlconf.notifications.smtp.config_secret_name
    if framework.utils.singletons.k8s.get_k8s_helper().is_running_inside_kubernetes_cluster():
        try:
            return framework.utils.singletons.k8s.get_k8s_helper().store_secrets(
                smtp_config_secret_name, secrets=default_params
            )
        except ApiException as exc:
            logger.warning(
                "Failed to store SMTP configuration secret",
                secret_name=smtp_config_secret_name,
                body=mlrun.errors.err_to_str(exc.body),
            )
