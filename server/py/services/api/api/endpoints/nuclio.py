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


@router.get(
    "/projects/{project}/api-gateways",
    response_model=mlrun.common.schemas.APIGatewaysOutput,
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def list_api_gateways(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    await server.py.services.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=mlrun.common.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )
    async with server.api.utils.clients.async_nuclio.Client(auth_info) as client:
        api_gateways = await client.list_api_gateways(project)

    allowed_api_gateways = await server.py.services.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.api_gateway,
        list(api_gateways.keys()) if api_gateways else [],
        lambda _api_gateway: (
            project,
            _api_gateway,
        ),
        auth_info,
    )
    allowed_api_gateways = {
        api_gateway: api_gateways[api_gateway] for api_gateway in allowed_api_gateways
    }
    return mlrun.common.schemas.APIGatewaysOutput(api_gateways=allowed_api_gateways)


@router.get(
    "/projects/{project}/api-gateways/{gateway}",
    response_model=mlrun.common.schemas.APIGateway,
    response_model_exclude_none=True,
)
async def get_api_gateway(
    project: str,
    gateway: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=mlrun.common.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.api_gateway,
        project,
        gateway,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    async with server.api.utils.clients.async_nuclio.Client(auth_info) as client:
        api_gateway = await client.get_api_gateway(project_name=project, name=gateway)

    return api_gateway


@router.put(
    "/projects/{project}/api-gateways/{gateway}",
    response_model=mlrun.common.schemas.APIGateway,
    response_model_exclude_none=True,
)
async def store_api_gateway(
    project: str,
    gateway: str,
    api_gateway: mlrun.common.schemas.APIGateway,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=mlrun.common.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.api_gateway,
        project,
        gateway,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    async with server.api.utils.clients.async_nuclio.Client(auth_info) as client:
        create = not await client.api_gateway_exists(
            name=gateway,
            project_name=project,
        )
        await client.store_api_gateway(
            project_name=project,
            api_gateway_name=gateway,
            api_gateway=api_gateway,
            create=create,
        )
        api_gateway = await client.get_api_gateway(
            name=gateway,
            project_name=project,
        )
    return api_gateway
