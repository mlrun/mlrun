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

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

import mlrun.common.schemas

import framework.service
import framework.utils.singletons.project_member
from framework.api import deps

router = APIRouter(prefix="/alert-templates")


@router.put("/{name}", response_model=mlrun.common.schemas.AlertTemplate)
@inject
async def store_alert_template(
    request: Request,
    name: str,
    alert_data: mlrun.common.schemas.AlertTemplate,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
    service: framework.service.Service = Depends(
        Provide[framework.service.ServiceContainer.service]
    ),
) -> mlrun.common.schemas.AlertTemplate:
    return await service.handle_request(
        "store_alert_template",
        request,
        name,
        alert_data,
        auth_info,
        db_session,
    )


@router.get(
    "/{name}",
    response_model=mlrun.common.schemas.AlertTemplate,
)
@inject
async def get_alert_template(
    request: Request,
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
    service: framework.service.Service = Depends(
        Provide[framework.service.ServiceContainer.service]
    ),
) -> mlrun.common.schemas.AlertTemplate:
    return await service.handle_request(
        "get_alert_template",
        request,
        name,
        auth_info,
        db_session,
    )


@router.get("", response_model=list[mlrun.common.schemas.AlertTemplate])
@inject
async def list_alert_templates(
    request: Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
    service: framework.service.Service = Depends(
        Provide[framework.service.ServiceContainer.service]
    ),
) -> list[mlrun.common.schemas.AlertTemplate]:
    return await service.handle_request(
        "list_alert_templates",
        request,
        auth_info,
        db_session,
    )


@router.delete(
    "/{name}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
@inject
async def delete_alert_template(
    request: Request,
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
    service: framework.service.Service = Depends(
        Provide[framework.service.ServiceContainer.service]
    ),
):
    return await service.handle_request(
        "delete_alert_template",
        request,
        name,
        auth_info,
        db_session,
    )
