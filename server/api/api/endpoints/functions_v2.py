# Copyright 2024 Iguazio
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
from fastapi import (
    APIRouter,
    Depends,
    Request,
)
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.model_monitoring
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas
import server.api.api.utils
import server.api.crud.model_monitoring.deployment
import server.api.crud.runtimes.nuclio.function
import server.api.db.session
import server.api.launcher
import server.api.utils.auth.verifier
import server.api.utils.background_tasks
import server.api.utils.clients.chief
import server.api.utils.functions
import server.api.utils.pagination
import server.api.utils.singletons.k8s
import server.api.utils.singletons.project_member
from mlrun.utils import logger
from server.api.api import deps
from server.api.utils.singletons.scheduler import get_scheduler

router = APIRouter()


@router.delete(
    "/projects/{project}/functions/{name}",
    responses={
        http.HTTPStatus.ACCEPTED.value: {"model": mlrun.common.schemas.BackgroundTask},
    },
)
async def delete_function(
    background_tasks: fastapi.BackgroundTasks,
    response: fastapi.Response,
    request: Request,
    project: str,
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    #  If the requested function has a schedule, we must delete it before deleting the function
    try:
        function_schedule = await run_in_threadpool(
            get_scheduler().get_schedule,
            db_session,
            project,
            name,
        )
    except mlrun.errors.MLRunNotFoundError:
        function_schedule = None

    if function_schedule:
        # when deleting a function, we should also delete its schedules if exists
        # schedules are only supposed to be run by the chief, therefore, if the function has a schedule,
        # and we are running in worker, we send the request to the chief client
        if (
            mlrun.mlconf.httpdb.clusterization.role
            != mlrun.common.schemas.ClusterizationRole.chief
        ):
            logger.info(
                "Function has a schedule, deleting",
                function=name,
                project=project,
            )
            chief_client = server.api.utils.clients.chief.Client()
            await chief_client.delete_schedule(
                project=project, name=name, request=request
            )
        else:
            await run_in_threadpool(
                get_scheduler().delete_schedule, db_session, project, name
            )
    task = await run_in_threadpool(
        server.api.api.utils.create_function_deletion_background_task,
        background_tasks,
        db_session,
        project,
        name,
        auth_info,
    )

    response.status_code = http.HTTPStatus.ACCEPTED.value
    return task
