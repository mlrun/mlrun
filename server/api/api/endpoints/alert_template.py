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

from http import HTTPStatus

import semver
from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.schemas
import server.api.utils.auth.verifier
import server.api.utils.singletons.project_member
from mlrun.utils import logger
from server.api.api import deps

router = APIRouter(prefix="/alert-templates")


@router.put("/{name}", response_model=mlrun.common.schemas.AlertTemplate)
async def store_alert_template(
    request: Request,
    name: str,
    alert_data: mlrun.common.schemas.AlertTemplate,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
) -> mlrun.common.schemas.AlertTemplate:
    await (
        server.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
            _get_authorization_resource(),
            mlrun.common.schemas.AuthorizationAction.create,
            auth_info,
        )
    )

    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        chief_client = server.api.utils.clients.chief.Client()
        data = await request.json()
        return await chief_client.store_alert_template(
            name=name, request=request, json=data
        )

    logger.debug("Storing alert template", name=name)

    return await run_in_threadpool(
        server.api.crud.AlertTemplates().store_alert_template,
        db_session,
        name,
        alert_data,
    )


@router.get(
    "/{name}",
    response_model=mlrun.common.schemas.AlertTemplate,
)
async def get_alert_template(
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
) -> mlrun.common.schemas.AlertTemplate:
    await (
        server.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
            _get_authorization_resource(),
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    )

    return await run_in_threadpool(
        server.api.crud.AlertTemplates().get_alert_template, db_session, name
    )


@router.get("", response_model=list[mlrun.common.schemas.AlertTemplate])
async def list_alert_templates(
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
) -> list[mlrun.common.schemas.AlertTemplate]:
    await (
        server.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
            _get_authorization_resource(),
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    )

    return await run_in_threadpool(
        server.api.crud.AlertTemplates().list_alert_templates, db_session
    )


@router.delete(
    "/{name}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
async def delete_alert_template(
    request: Request,
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await (
        server.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
            _get_authorization_resource(),
            mlrun.common.schemas.AuthorizationAction.delete,
            auth_info,
        )
    )

    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        chief_client = server.api.utils.clients.chief.Client()
        return await chief_client.delete_alert_template(name=name, request=request)

    logger.debug("Deleting alert template", name=name)

    await run_in_threadpool(
        server.api.crud.AlertTemplates().delete_alert_template, db_session, name
    )


def _get_authorization_resource():
    igz_version = mlrun.mlconf.get_parsed_igz_version()
    if igz_version and igz_version < semver.VersionInfo.parse("3.6.0"):
        # alert_templates is not in OFA manifest prior to 3.6, so we use
        # the permissions of hub_source as they are the same
        return mlrun.common.schemas.AuthorizationResourceTypes.hub_source

    return mlrun.common.schemas.AuthorizationResourceTypes.alert_templates
