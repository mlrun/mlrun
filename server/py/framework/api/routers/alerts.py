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

import framework.api.deps
import framework.service
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

import mlrun.common.schemas

router = APIRouter(prefix="/projects/{project}/alerts")


@router.put("/{name}", response_model=mlrun.common.schemas.AlertConfig)
@inject
async def store_alert(
    request: Request,
    project: str,
    name: str,
    alert_data: mlrun.common.schemas.AlertConfig,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(
        Provide[framework.api.deps.DepsContainer.authenticate_request]
    ),
    db_session: Session = Depends(Provide[framework.api.deps.DepsContainer.db_session]),
    service: framework.service.Service = Depends(
        Provide[framework.service.ServiceContainer.service]
    ),
) -> mlrun.common.schemas.AlertConfig:
    return await service.handle_request(
        "/projects/{project}/alerts/{name}",
        request,  # has method inside
        project,
        name,
        alert_data,
        auth_info,
        db_session,
    )
