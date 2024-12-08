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


from typing import Optional, Union

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

import mlrun.common.schemas

import framework.utils.auth.verifier
import framework.utils.clients.chief
import framework.utils.singletons.project_member
from framework.api import deps

router = APIRouter()


@router.get("/projects/{project}/alerts/{name}/activations")
@router.get("/projects/{project}/alert-activations")
@inject
async def list_alert_activations(
    request: Request,
    project: str,
    name: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    entity: Optional[str] = None,
    severity: Optional[
        list[Union[mlrun.common.schemas.alert.AlertSeverity, str]]
    ] = Query([], alias="severity"),
    entity_kind: Optional[
        Union[mlrun.common.schemas.alert.EventEntityKind, str]
    ] = Query(None, alias="entity-kind"),
    event_kind: Optional[Union[mlrun.common.schemas.alert.EventKind, str]] = Query(
        None, alias="event-kind"
    ),
    page: int = Query(None, gt=0),
    page_size: int = Query(None, alias="page-size", gt=0),
    page_token: str = Query(None, alias="page-token"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
    service: framework.service.Service = Depends(
        Provide[framework.service.ServiceContainer.service]
    ),
):
    return await service.handle_request(
        "list_alert_activations",
        request=request,
        project=project,
        name=name,
        since=since,
        until=until,
        entity=entity,
        severity=severity,
        entity_kind=entity_kind,
        event_kind=event_kind,
        page=page,
        page_size=page_size,
        page_token=page_token,
        auth_info=auth_info,
        db_session=db_session,
    )
