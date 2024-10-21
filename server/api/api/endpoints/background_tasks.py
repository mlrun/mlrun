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
import datetime
import typing

import fastapi
import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas
import mlrun.utils
import server.api.api.deps
import server.api.utils.auth.verifier
import server.api.utils.background_tasks
import server.api.utils.clients.chief
from mlrun.utils import logger

router = fastapi.APIRouter()


@router.get(
    "/projects/{project}/background-tasks/{name}",
    response_model=mlrun.common.schemas.BackgroundTask,
)
async def get_project_background_task(
    project: str,
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
):
    # Since there's no not-found option on get_project_background_task - we authorize before getting (unlike other
    # get endpoint)
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.project_background_task,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return await run_in_threadpool(
        server.api.utils.background_tasks.ProjectBackgroundTasksHandler().get_background_task,
        db_session,
        name=name,
        project=project,
    )


@router.get(
    "/projects/{project}/background-tasks",
    response_model=mlrun.common.schemas.BackgroundTaskList,
)
async def list_project_background_tasks(
    project: str,
    state: str = None,
    created_from: str = None,
    created_to: str = None,
    last_update_time_from: str = None,
    last_update_time_to: str = None,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    if (
        not state
        and not created_from
        and not created_to
        and not last_update_time_from
        and not last_update_time_to
    ):
        # default to last week on no filter
        created_from = (
            datetime.datetime.now() - datetime.timedelta(days=7)
        ).isoformat()

    background_tasks = await run_in_threadpool(
        server.api.utils.background_tasks.ProjectBackgroundTasksHandler().list_background_tasks,
        db_session,
        project=project,
        states=[state] if state is not None else None,
        created_from=mlrun.utils.datetime_from_iso(created_from),
        created_to=mlrun.utils.datetime_from_iso(created_to),
        last_update_time_from=mlrun.utils.datetime_from_iso(last_update_time_from),
        last_update_time_to=mlrun.utils.datetime_from_iso(last_update_time_to),
    )

    background_tasks = await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.project_background_task,
        background_tasks,
        lambda background_task: (
            background_task.metadata.project,
            background_task.metadata.name,
        ),
        auth_info,
    )

    return mlrun.common.schemas.BackgroundTaskList(background_tasks=background_tasks)


@router.get(
    "/background-tasks/{name}",
    response_model=mlrun.common.schemas.BackgroundTask,
)
async def get_internal_background_task(
    name: str,
    request: fastapi.Request,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
):
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting internal background task, re-routing to chief",
            internal_background_task=name,
        )
        chief_client = server.api.utils.clients.chief.Client()
        return await chief_client.get_internal_background_task(
            name=name, request=request
        )

    background_task = await run_in_threadpool(
        server.api.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task,
        name=name,
        raise_on_not_found=True,
    )
    await _authorize_get_background_task_request(background_task, auth_info)
    return background_task


@router.get(
    "/background-tasks",
    response_model=mlrun.common.schemas.BackgroundTaskList,
)
async def list_internal_background_tasks(
    request: fastapi.Request,
    name: typing.Optional[str] = None,
    kind: typing.Optional[str] = None,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
):
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting internal background tasks, re-routing to chief",
            internal_background_task=name,
        )
        chief_client = server.api.utils.clients.chief.Client()
        return await chief_client.get_internal_background_tasks(request=request)

    background_tasks = server.api.utils.background_tasks.InternalBackgroundTasksHandler().list_background_tasks(
        name=name,
        kind=kind,
    )

    allowed_background_tasks = []
    for background_task in background_tasks:
        try:
            await _authorize_get_background_task_request(background_task, auth_info)
            allowed_background_tasks.append(background_task)
        except mlrun.errors.MLRunAccessDeniedError:
            pass

    return mlrun.common.schemas.BackgroundTaskList(
        background_tasks=allowed_background_tasks
    )


async def _authorize_get_background_task_request(
    background_task: mlrun.common.schemas.BackgroundTask,
    auth_info: mlrun.common.schemas.AuthInfo,
):
    return

    # TODO: Check resource permissions for background tasks. We need to ensure that the user can read the project
    # when attempting to access the background task. It appears that when a project is deleted, its associated
    # policies are also removed, leading to "access denied" errors for users trying to access the background task.
    # Related to ML-7484
    # Change the test `test_get_internal_background_task_auth` once this is resolved.

    # Iguazio manifest doesn't support the global background task resource yet - therefore if the background task has a
    # project (e.g. delete project), we can authorize on the project
    # if background_task.metadata.project:
    #     return await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
    #         background_task.metadata.project,
    #         mlrun.common.schemas.AuthorizationAction.read,
    #         auth_info,
    #     )

    # If there is no project we have to just omit authorization until iguazio supports it
    # igz_version = mlrun.mlconf.get_parsed_igz_version()
    # if igz_version and igz_version >= semver.VersionInfo.parse("3.7.0-b1"):
    #     return await server.api.utils.auth.verifier.AuthVerifier().query_resource_permissions(
    #         mlrun.common.schemas.AuthorizationResourceTypes.background_task,
    #         background_task.metadata.name,
    #         mlrun.common.schemas.AuthorizationAction.read,
    #         auth_info,
    #     )
