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
import asyncio
import base64
import typing

import fastapi

import mlrun
import mlrun.api.schemas
import mlrun.api.utils.auth.providers.nop
import mlrun.api.utils.auth.providers.opa
import mlrun.api.utils.clients.iguazio
import mlrun.utils.singleton


class AuthVerifier(metaclass=mlrun.utils.singleton.Singleton):
    _basic_prefix = "Basic "
    _bearer_prefix = "Bearer "

    def __init__(self) -> None:
        super().__init__()
        if mlrun.mlconf.httpdb.authorization.mode == "none":
            self._auth_provider = mlrun.api.utils.auth.providers.nop.Provider()
        elif mlrun.mlconf.httpdb.authorization.mode == "opa":
            self._auth_provider = mlrun.api.utils.auth.providers.opa.Provider()
        else:
            raise NotImplementedError("Unsupported authorization mode")

    async def filter_project_resources_by_permissions(
        self,
        resource_type: mlrun.api.schemas.AuthorizationResourceTypes,
        resources: typing.List,
        project_and_resource_name_extractor: typing.Callable,
        auth_info: mlrun.api.schemas.AuthInfo,
        action: mlrun.api.schemas.AuthorizationAction = mlrun.api.schemas.AuthorizationAction.read,
    ) -> typing.List:
        def _generate_opa_resource(resource):
            project_name, resource_name = project_and_resource_name_extractor(resource)
            return self._generate_resource_string_from_project_resource(
                resource_type, project_name, resource_name
            )

        return await self.filter_by_permissions(
            resources, _generate_opa_resource, action, auth_info
        )

    async def filter_projects_by_permissions(
        self,
        project_names: typing.List[str],
        auth_info: mlrun.api.schemas.AuthInfo,
        action: mlrun.api.schemas.AuthorizationAction = mlrun.api.schemas.AuthorizationAction.read,
    ) -> typing.List:
        return await self.filter_by_permissions(
            project_names,
            self._generate_resource_string_from_project_name,
            action,
            auth_info,
        )

    async def query_project_resources_permissions(
        self,
        resource_type: mlrun.api.schemas.AuthorizationResourceTypes,
        resources: typing.List,
        project_and_resource_name_extractor: typing.Callable,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        project_resources = [
            # project name, resource name
            project_and_resource_name_extractor(resource)
            for resource in resources
        ]
        return all(
            await asyncio.gather(
                *[
                    self.query_project_resource_permissions(
                        resource_type,
                        project_resource[0],
                        project_resource[1],
                        action,
                        auth_info,
                        raise_on_forbidden,
                    )
                    for project_resource in project_resources
                ]
            )
        )

    async def query_project_resource_permissions(
        self,
        resource_type: mlrun.api.schemas.AuthorizationResourceTypes,
        project_name: str,
        resource_name: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        return await self.query_permissions(
            self._generate_resource_string_from_project_resource(
                resource_type, project_name, resource_name
            ),
            action,
            auth_info,
            raise_on_forbidden,
        )

    async def query_project_permissions(
        self,
        project_name: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        return await self.query_permissions(
            self._generate_resource_string_from_project_name(project_name),
            action,
            auth_info,
            raise_on_forbidden,
        )

    async def query_global_resource_permissions(
        self,
        resource_type: mlrun.api.schemas.AuthorizationResourceTypes,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        return await self.query_resource_permissions(
            resource_type,
            "",
            action,
            auth_info,
            raise_on_forbidden,
        )

    async def query_resource_permissions(
        self,
        resource_type: mlrun.api.schemas.AuthorizationResourceTypes,
        resource_name: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        return await self.query_permissions(
            resource_type.to_resource_string("", resource_name),
            action=action,
            auth_info=auth_info,
            raise_on_forbidden=raise_on_forbidden,
        )

    async def query_permissions(
        self,
        resource: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        return await self._auth_provider.query_permissions(
            resource, action, auth_info, raise_on_forbidden
        )

    async def filter_by_permissions(
        self,
        resources: typing.List,
        opa_resource_extractor: typing.Callable,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
    ) -> typing.List:
        return await self._auth_provider.filter_by_permissions(
            resources,
            opa_resource_extractor,
            action,
            auth_info,
        )

    def add_allowed_project_for_owner(
        self, project_name: str, auth_info: mlrun.api.schemas.AuthInfo
    ):
        self._auth_provider.add_allowed_project_for_owner(project_name, auth_info)

    async def authenticate_request(
        self, request: fastapi.Request
    ) -> mlrun.api.schemas.AuthInfo:
        auth_info = mlrun.api.schemas.AuthInfo()
        header = request.headers.get("Authorization", "")
        if self._basic_auth_configured():
            if not header.startswith(self._basic_prefix):
                raise mlrun.errors.MLRunUnauthorizedError("Missing basic auth header")
            username, password = self._parse_basic_auth(header)
            if (
                username != mlrun.mlconf.httpdb.authentication.basic.username
                or password != mlrun.mlconf.httpdb.authentication.basic.password
            ):
                raise mlrun.errors.MLRunUnauthorizedError(
                    "Username or password did not match"
                )
            auth_info.username = username
            auth_info.password = password
        elif self._bearer_auth_configured():
            if not header.startswith(self._bearer_prefix):
                raise mlrun.errors.MLRunUnauthorizedError("Missing bearer auth header")
            token = header[len(self._bearer_prefix) :]
            if token != mlrun.mlconf.httpdb.authentication.bearer.token:
                raise mlrun.errors.MLRunUnauthorizedError("Token did not match")
            auth_info.token = token
        elif self._iguazio_auth_configured():
            iguazio_client = mlrun.api.utils.clients.iguazio.AsyncClient()
            auth_info = await iguazio_client.verify_request_session(request)
            if "x-data-session-override" in request.headers:
                auth_info.data_session = request.headers["x-data-session-override"]

        # Fallback in case auth method didn't fill in the username already, and it is provided by the caller
        if not auth_info.username and "x-remote-user" in request.headers:
            auth_info.username = request.headers["x-remote-user"]

        projects_role_header = request.headers.get(
            mlrun.api.schemas.HeaderNames.projects_role
        )
        auth_info.projects_role = (
            mlrun.api.schemas.ProjectsRole(projects_role_header)
            if projects_role_header
            else None
        )
        # In Iguazio 3.0 we're running with auth mode none cause auth is done by the ingress, in that auth mode sessions
        # needed for data operations were passed through this header, keep reading it to be backwards compatible
        if not auth_info.data_session and "X-V3io-Session-Key" in request.headers:
            auth_info.data_session = request.headers["X-V3io-Session-Key"]
        # In Iguazio 3.0 the ingress auth verification overrides the X-V3io-Session-Key from the auth response
        # therefore the above won't work for requests coming from outside the cluster so allowing another header that
        # won't be overridden
        if not auth_info.data_session and "X-V3io-Access-Key" in request.headers:
            auth_info.data_session = request.headers["X-V3io-Access-Key"]
        return auth_info

    async def generate_auth_info_from_session(
        self, session: str
    ) -> mlrun.api.schemas.AuthInfo:
        if not self._iguazio_auth_configured():
            raise NotImplementedError(
                "Session is currently supported only for iguazio authentication mode"
            )
        return await mlrun.api.utils.clients.iguazio.AsyncClient().verify_session(
            session
        )

    def get_or_create_access_key(
        self, session: str, planes: typing.List[str] = None
    ) -> str:
        if not self._iguazio_auth_configured():
            raise NotImplementedError(
                "Access key is currently supported only for iguazio authentication mode"
            )
        return mlrun.api.utils.clients.iguazio.Client().get_or_create_access_key(
            session, planes
        )

    def is_jobs_auth_required(self):
        return self._iguazio_auth_configured()

    @staticmethod
    def _generate_resource_string_from_project_name(project_name: str):
        return mlrun.api.schemas.AuthorizationResourceTypes.project.to_resource_string(
            project_name, ""
        )

    @staticmethod
    def _generate_resource_string_from_project_resource(
        resource_type: mlrun.api.schemas.AuthorizationResourceTypes,
        project_name: str,
        resource_name: str,
    ):
        if not project_name:
            project_name = "*"
        if not resource_name:
            resource_name = "*"
        return resource_type.to_resource_string(project_name, resource_name)

    @staticmethod
    def _basic_auth_configured():
        return mlrun.mlconf.httpdb.authentication.mode == "basic" and (
            mlrun.mlconf.httpdb.authentication.basic.username
            or mlrun.mlconf.httpdb.authentication.basic.password
        )

    @staticmethod
    def _bearer_auth_configured():
        return (
            mlrun.mlconf.httpdb.authentication.mode == "bearer"
            and mlrun.mlconf.httpdb.authentication.bearer.token
        )

    @staticmethod
    def _iguazio_auth_configured():
        return mlrun.mlconf.httpdb.authentication.mode == "iguazio"

    @staticmethod
    def _parse_basic_auth(header):
        """
        parse_basic_auth('Basic YnVnczpidW5ueQ==')
        ['bugs', 'bunny']
        """
        b64value = header[len(AuthVerifier._basic_prefix) :]
        value = base64.b64decode(b64value).decode()
        return value.split(":", 1)
