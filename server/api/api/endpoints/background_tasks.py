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
import fastapi
import semver
import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas
import server.api.api.deps
import server.api.utils.auth.verifier
import server.api.utils.background_tasks
import server.api.utils.clients.internal
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
    # Since there's no not-found option on get_background_task - we authorize before getting (unlike other get endpoint)
    # In Iguazio 3.2 the manifest doesn't support the global background task resource - therefore we have to just omit
    # authorization
    # we also skip Iguazio 3.6 for now, until it will add support for it (still in development)
    igz_version = mlrun.mlconf.get_parsed_igz_version()
    if igz_version and igz_version >= semver.VersionInfo.parse("3.7.0-b1"):
        await server.api.utils.auth.verifier.AuthVerifier().query_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.background_task,
            name,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting internal background task, re-routing to chief",
            internal_background_task=name,
        )
        chief_client = server.api.utils.clients.internal.Client()
        return await chief_client.get_internal_background_task(
            name=name, request=request
        )

    return await run_in_threadpool(
        server.api.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task,
        name=name,
    )
