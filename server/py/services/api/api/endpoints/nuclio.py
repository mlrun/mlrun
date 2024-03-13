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
from fastapi import APIRouter, Depends

import mlrun.common.schemas
import server.py.services.api.utils.clients.async_nuclio
from server.py.services.api.api import deps

router = APIRouter()


@router.get("/projects/{project}/nuclio/api-gateways")
async def list_api_gateways(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    await server.py.services.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=mlrun.common.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )

    api_gateways = await server.py.services.api.utils.clients.async_nuclio.Client(
        auth_info
    ).list_api_gateways(project)

    return await server.py.services.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.api_gateways,
        list(api_gateways.values()) if api_gateways else [],
        lambda _api_gateway: (
            _api_gateway.get("metadata", {})
            .get("labels", {})
            .get("iguazio.com/username"),
            _api_gateway.get("metadata", {}).get("name"),
        ),
        auth_info,
    )
