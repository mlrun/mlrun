# Copyright 2018 Iguazio
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
import fastapi.concurrency
import sqlalchemy.orm

import mlrun.api.api.deps
import mlrun.api.crud.notifications
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.clients.chief
import mlrun.api.utils.singletons.project_member
import mlrun.common.schemas
from mlrun.utils import logger

router = fastapi.APIRouter(prefix="/projects/{project}/notifications")


CHIEF_REDIRECTED_NOTIFICATIONS = [
    "schedule",
]


@router.put("", status_code=http.HTTPStatus.ACCEPTED.value)
async def set_object_notifications(
    project: str,
    request: fastapi.Request,
    set_notifications_request: mlrun.common.schemas.SetNotificationRequest = fastapi.Body(
        ...
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    # check permission per object type
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        getattr(
            mlrun.common.schemas.AuthorizationResourceTypes,
            set_notifications_request.parent.identifier.kind,
        ),
        project,
        resource_name="notifications",
        action=mlrun.common.schemas.AuthorizationAction.update,
        auth_info=auth_info,
    )

    if (
        set_notifications_request.parent.identifier.kind
        in CHIEF_REDIRECTED_NOTIFICATIONS
        and mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to set schedule notifications, re-routing to chief",
            project=project,
            schedule=set_notifications_request.dict(),
        )
        chief_client = mlrun.api.utils.clients.chief.Client()
        return await chief_client.set_object_notifications(
            project=project,
            request=request,
            json=set_notifications_request.dict(),
        )

    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.crud.Notifications().set_object_notifications,
        db_session,
        auth_info,
        project,
        set_notifications_request.notifications,
        set_notifications_request.parent,
    )
    return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)
