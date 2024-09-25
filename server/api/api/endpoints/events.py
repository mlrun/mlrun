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

from http import HTTPStatus

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.schemas
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.clients.chief
import server.api.utils.singletons.project_member
from mlrun.utils import logger
from server.api.api import deps

router = APIRouter()


@router.post("/projects/{project}/events/{name}")
async def post_event(
    request: Request,
    project: str,
    name: str,
    event_data: mlrun.common.schemas.Event,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.event,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )

    if mlrun.mlconf.alerts.mode == mlrun.common.schemas.alert.AlertsModes.disabled:
        logger.debug(
            "Alerts are disabled, skipping event processing",
            project=project,
            event_name=name,
        )
        return

    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        data = await request.json()
        chief_client = server.api.utils.clients.chief.Client()
        return await chief_client.set_event(
            project=project, name=name, request=request, json=data
        )

    logger.debug("Got event", project=project, name=name, id=event_data.entity.ids[0])

    if not server.api.crud.Events().is_valid_event(project, event_data):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value)

    await run_in_threadpool(
        server.api.crud.Events().process_event, db_session, event_data, name, project
    )
