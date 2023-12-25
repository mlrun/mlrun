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
import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas
import server.api.api.deps
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.background_tasks
import server.api.utils.clients.chief
import server.api.utils.helpers
from mlrun.utils import logger
from server.api.utils.singletons.project_member import get_project_member

router = fastapi.APIRouter()


class Constants:
    project_deletion_wrapper_background_task_name_format = (
        "project_deletion_wrapper_{0}"
    )
    project_deletion_background_task_name_format = "project_deletion_{0}"


@router.delete(
    "/projects/{name}",
    responses={
        http.HTTPStatus.NO_CONTENT.value: {},
        http.HTTPStatus.ACCEPTED.value: {"model": mlrun.common.schemas.BackgroundTask},
    },
)
async def delete_project(
    background_tasks: fastapi.BackgroundTasks,
    response: fastapi.Response,
    request: fastapi.Request,
    name: str,
    deletion_strategy: mlrun.common.schemas.DeletionStrategy = fastapi.Header(
        mlrun.common.schemas.DeletionStrategy.default(),
        alias=mlrun.common.schemas.HeaderNames.deletion_strategy,
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
):
    # delete project can be responsible for deleting schedules. Schedules are running only on chief,
    # that is why we re-route requests to chief
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to delete project, re-routing to chief",
            project=name,
            deletion_strategy=deletion_strategy,
        )
        chief_client = server.api.utils.clients.chief.Client()
        return await chief_client.delete_project(
            name=name, request=request, api_version="v2"
        )

    # check if project exists
    try:
        get_project_member().get_project(db_session, name, auth_info.session)
    except mlrun.errors.MLRunNotFoundError:
        logger.info("Project not found, nothing to delete", project=name)
        return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)

    # as opposed to v1, we need to implement the `check` deletion strategy here, since we don't want
    # to spawn a background task for this, only to return a response
    if (
        server.api.utils.helpers.is_request_from_leader(auth_info.projects_role)
        and deletion_strategy == mlrun.common.schemas.DeletionStrategy.check
    ):
        response.status_code = http.HTTPStatus.NO_CONTENT.value
        return server.api.crud.Projects().verify_project_is_empty(db_session, name)

    background_task = await run_in_threadpool(
        _get_or_create_project_deletion_background_task,
        name,
        deletion_strategy,
        background_tasks,
        db_session,
        auth_info,
    )
    response.status_code = http.HTTPStatus.ACCEPTED.value
    return background_task


def _get_or_create_project_deletion_background_task(
    project_name: str, deletion_strategy: str, background_tasks, db_session, auth_info
) -> typing.Optional[mlrun.common.schemas.BackgroundTask]:
    """
    This method is responsible for creating a background task for deleting a project.
    The project deletion flow is as follows:

        When MLRun is leader:
        1. Create a background task for deleting the project
        2. The background task will delete the project resources and then project itself

        When MLRun is a follower:
        Due to the nature of the project deletion flow, we need to wrap the task as:
            MLRunDeletionWrapperTask(LeaderDeletionJob(MLRunDeletionTask))

        1. Create MLRunDeletionWrapperTask
        2. MLRunDeletionWrapperTask will send a request to the projects leader to delete the project
        3. MLRunDeletionWrapperTask will wait for the project to be deleted using LeaderDeletionJob job id
           During (In leader):
           1. Create LeaderDeletionJob
           2. LeaderDeletionJob will send a second delete project request to the follower
           3. LeaderDeletionJob will wait for the project to be deleted using the MLRunDeletionTask task id
              During (Back here in follower):
              1. Create MLRunDeletionTask
              2. MLRunDeletionTask will delete the project resources and then project itself.
              3. Finish MLRunDeletionTask
           4. Finish LeaderDeletionJob
        4. Finish MLRunDeletionWrapperTask
    """

    # If the request is from the leader, or MLRun is the leader, we create a background task for deleting the
    # project. Otherwise, we create a wrapper background task for deletion of the project.
    background_task_name_format = (
        Constants.project_deletion_wrapper_background_task_name_format
    )
    if (
        server.api.utils.helpers.is_request_from_leader(auth_info.projects_role)
        or mlrun.mlconf.httpdb.projects.leader == "mlrun"
    ):
        background_task_name_format = (
            Constants.project_deletion_background_task_name_format
        )

    background_task_name = background_task_name_format.format(project_name)
    try:
        return server.api.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task(
            background_task_name,
            raise_on_not_found=True,
        )
    except mlrun.errors.MLRunNotFoundError:
        return server.api.utils.background_tasks.InternalBackgroundTasksHandler().create_background_task(
            background_tasks,
            _delete_project,
            name=background_task_name,
            db_session=db_session,
            project_name=project_name,
            deletion_strategy=deletion_strategy,
            auth_info=auth_info,
        )


async def _delete_project(
    db_session: sqlalchemy.orm.Session,
    project_name: str,
    deletion_strategy: mlrun.common.schemas.DeletionStrategy,
    auth_info: mlrun.common.schemas.AuthInfo,
):
    await run_in_threadpool(
        get_project_member().delete_project,
        db_session,
        project_name,
        deletion_strategy,
        auth_info.projects_role,
        auth_info,
        wait_for_completion=True,
    )
    await get_project_member().post_delete_project(project_name)
