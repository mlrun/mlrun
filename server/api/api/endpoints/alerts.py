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

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.schemas
import mlrun.common.schemas.alert
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.clients.chief
import server.api.utils.singletons.project_member
from mlrun.utils import logger
from server.api.api import deps

router = APIRouter(prefix="/projects/{project}/alerts")


@router.put("/{name}", response_model=mlrun.common.schemas.AlertConfig)
async def store_alert(
    request: Request,
    project: str,
    name: str,
    alert_data: mlrun.common.schemas.AlertConfig,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
) -> mlrun.common.schemas.AlertConfig:
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.alert,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.create,
        auth_info,
    )

    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        chief_client = server.api.utils.clients.chief.Client()
        data = await request.json()
        return await chief_client.store_alert(
            project=project, name=name, request=request, json=data
        )

    logger.debug("Storing alert", project=project, name=name)

    return await run_in_threadpool(
        server.api.crud.Alerts().store_alert,
        db_session,
        project,
        name,
        alert_data,
    )


@router.get(
    "/{name}",
    response_model=mlrun.common.schemas.AlertConfig,
)
async def get_alert(
    project: str,
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
) -> mlrun.common.schemas.AlertConfig:
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.alert,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    return await run_in_threadpool(
        server.api.crud.Alerts().get_enriched_alert, db_session, project, name
    )


@router.get("", response_model=list[mlrun.common.schemas.AlertConfig])
async def list_alerts(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
) -> list[mlrun.common.schemas.AlertConfig]:
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    alerts = await run_in_threadpool(
        server.api.crud.Alerts().list_alerts, db_session, project
    )

    alerts = await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.alert,
        alerts,
        lambda alert: (
            alert.project,
            alert.name,
        ),
        auth_info,
    )

    return alerts


@router.delete(
    "/{name}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
async def delete_alert(
    request: Request,
    project: str,
    name: str,
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
        mlrun.common.schemas.AuthorizationResourceTypes.alert,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )

    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        chief_client = server.api.utils.clients.chief.Client()
        return await chief_client.delete_alert(
            project=project, name=name, request=request
        )

    logger.debug("Deleting alert", project=project, name=name)

    await run_in_threadpool(
        server.api.crud.Alerts().delete_alert, db_session, project, name
    )


@router.post("/{name}/reset")
async def reset_alert(
    request: Request,
    project: str,
    name: str,
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
        mlrun.common.schemas.AuthorizationResourceTypes.alert,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.update,
        auth_info,
    )

    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        chief_client = server.api.utils.clients.chief.Client()
        return await chief_client.reset_alert(
            project=project, name=name, request=request
        )

    logger.debug("Resetting alert", project=project, name=name)

    return await run_in_threadpool(
        server.api.crud.Alerts().reset_alert, db_session, project, name
    )
