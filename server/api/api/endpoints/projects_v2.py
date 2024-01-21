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

import fastapi
import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas
import server.api.api.deps
import server.api.api.utils
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.background_tasks
import server.api.utils.clients.chief
import server.api.utils.helpers
from mlrun.utils import logger
from server.api.utils.singletons.project_member import get_project_member

router = fastapi.APIRouter()


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
    # check if project exists
    try:
        await run_in_threadpool(
            get_project_member().get_project, db_session, name, auth_info.session
        )
    except mlrun.errors.MLRunNotFoundError:
        logger.info("Project not found, nothing to delete", project=name)
        return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)

    # usually the CRUD for delete project will check permissions, however, since we are running the crud in a background
    # task, we need to check permissions here. skip permission check if the request is from the leader.
    if not server.api.utils.helpers.is_request_from_leader(auth_info.projects_role):
        await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
            name,
            mlrun.common.schemas.AuthorizationAction.delete,
            auth_info,
        )

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

    # as opposed to v1, we need to implement the `check` deletion strategy here, since we don't want
    # to spawn a background task for this, only to return a response
    if (
        server.api.utils.helpers.is_request_from_leader(auth_info.projects_role)
        and deletion_strategy == mlrun.common.schemas.DeletionStrategy.check
    ):
        response.status_code = http.HTTPStatus.NO_CONTENT.value
        return server.api.crud.Projects().verify_project_is_empty(db_session, name)

    background_task = await run_in_threadpool(
        server.api.api.utils.get_or_create_project_deletion_background_task,
        name,
        deletion_strategy,
        background_tasks,
        db_session,
        auth_info,
    )
    response.status_code = http.HTTPStatus.ACCEPTED.value
    return background_task
