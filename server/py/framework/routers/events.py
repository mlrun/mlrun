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


from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

import mlrun.common.schemas

import framework.service
from framework.api import deps

router = APIRouter()


@router.post("/projects/{project}/events/{name}")
@inject
async def process_event(
    request: Request,
    project: str,
    name: str,
    event_data: mlrun.common.schemas.Event,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
    service: framework.service.Service = Depends(
        Provide[framework.service.ServiceContainer.service]
    ),
):
    return await service.handle_request(
        "process_event",
        request,
        project,
        name,
        event_data,
        auth_info,
        db_session,
    )
