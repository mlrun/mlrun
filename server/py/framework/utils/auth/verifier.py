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

import asyncio
import base64
import time
import typing
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from heapq import heapify, heappop, heappush

import fastapi
import jwt

import mlrun
import mlrun.common.schemas as schemas
import mlrun.utils.helpers
import mlrun.utils.singleton
from mlrun.common.types import AuthenticationMode
from mlrun.utils import logger

import framework.utils.auth.providers.nop
import framework.utils.auth.providers.opa
import framework.utils.clients.iguazio.v3
import framework.utils.clients.iguazio.v4

TOKEN_CACHE_MAX_SIZE = 128
TOKEN_CACHE_MAX_TTL = 30  # seconds


@dataclass(frozen=True)
class TokenExpiryEntry:
    token: str
    task: asyncio.Task[schemas.AuthInfo]
    expires_at: float

    def __lt__(self, other: "TokenExpiryEntry") -> bool:
        return self.expires_at < other.expires_at


class AuthVerifier(metaclass=mlrun.utils.singleton.Singleton):
    _token_cache: OrderedDict[str, asyncio.Task[schemas.AuthInfo]]
    _token_expiry_heap: list[TokenExpiryEntry]

    def __init__(self) -> None:
        super().__init__()
        self._resources_prefix = mlrun.mlconf.httpdb.authorization.namespaces.resources
        self._mgmt_prefix = mlrun.mlconf.httpdb.authorization.namespaces.mgmt
        self._prefixes = {
            schemas.AuthorizationResourceNamespace.resources: self._resources_prefix,
            schemas.AuthorizationResourceNamespace.mgmt: self._mgmt_prefix,
        }
        if mlrun.mlconf.httpdb.authorization.mode == "none":
            self._auth_provider = framework.utils.auth.providers.nop.Provider()
        elif mlrun.mlconf.httpdb.authorization.mode == "opa":
            self._auth_provider = framework.utils.auth.providers.opa.Provider()
        else:
            raise NotImplementedError("Unsupported authorization mode")

        self._token_cache = OrderedDict()
        self._token_expiry_heap = []

    async def filter_project_resources_by_permissions(
        self,
        resource_type: schemas.AuthorizationResourceTypes,
        resources: list,
        project_and_resource_name_extractor: typing.Callable,
        auth_info: schemas.AuthInfo,
        action: schemas.AuthorizationAction = schemas.AuthorizationAction.read,
        resource_namespace: schemas.AuthorizationResourceNamespace = schemas.AuthorizationResourceNamespace.resources,
    ) -> list:
        def _generate_opa_resource(resource):
            project_name, resource_name = project_and_resource_name_extractor(resource)
            return self._generate_resource_string_from_project_resource(
                resource_type=resource_type,
                project_name=project_name,
                resource_name=resource_name,
                resource_namespace=resource_namespace,
            )

        return await self.filter_by_permissions(
            resources, _generate_opa_resource, action, auth_info
        )

    async def filter_projects_by_permissions(
        self,
        project_names: list[str],
        auth_info: schemas.AuthInfo,
        action: schemas.AuthorizationAction = schemas.AuthorizationAction.read,
        resource_namespace: schemas.AuthorizationResourceNamespace = schemas.AuthorizationResourceNamespace.resources,
    ) -> list:
        def _generate_project_resource(project):
            return self._generate_resource_string_from_project_name(
                project, resource_namespace
            )

        return await self.filter_by_permissions(
            project_names,
            _generate_project_resource,
            action,
            auth_info,
        )

    async def query_project_resources_permissions(
        self,
        resource_type: schemas.AuthorizationResourceTypes,
        resources: list,
        project_and_resource_name_extractor: typing.Callable,
        action: schemas.AuthorizationAction,
        auth_info: schemas.AuthInfo,
        raise_on_forbidden: bool = True,
        resource_namespace: schemas.AuthorizationResourceNamespace = schemas.AuthorizationResourceNamespace.resources,
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
                        resource_namespace,
                    )
                    for project_resource in project_resources
                ]
            )
        )

    async def query_project_resource_permissions(
        self,
        resource_type: schemas.AuthorizationResourceTypes,
        project_name: str,
        resource_name: str,
        action: schemas.AuthorizationAction,
        auth_info: schemas.AuthInfo,
        raise_on_forbidden: bool = True,
        resource_namespace: schemas.AuthorizationResourceNamespace = schemas.AuthorizationResourceNamespace.resources,
    ) -> bool:
        return await self.query_permissions(
            self._generate_resource_string_from_project_resource(
                resource_type=resource_type,
                project_name=project_name,
                resource_name=resource_name,
                resource_namespace=resource_namespace,
            ),
            action,
            auth_info,
            raise_on_forbidden,
        )

    async def query_project_permissions(
        self,
        project_name: str,
        action: schemas.AuthorizationAction,
        auth_info: schemas.AuthInfo,
        raise_on_forbidden: bool = True,
        resource_namespace: schemas.AuthorizationResourceNamespace = schemas.AuthorizationResourceNamespace.resources,
    ) -> bool:
        return await self.query_permissions(
            self._generate_resource_string_from_project_name(
                project_name, resource_namespace
            ),
            action,
            auth_info,
            raise_on_forbidden,
        )

    async def query_global_resource_permissions(
        self,
        resource_type: schemas.AuthorizationResourceTypes,
        action: schemas.AuthorizationAction,
        auth_info: schemas.AuthInfo,
        raise_on_forbidden: bool = True,
        resource_namespace: schemas.AuthorizationResourceNamespace = schemas.AuthorizationResourceNamespace.resources,
    ) -> bool:
        return await self.query_resource_permissions(
            resource_type,
            "",
            action,
            auth_info,
            raise_on_forbidden,
            resource_namespace,
        )

    async def query_resource_permissions(
        self,
        resource_type: schemas.AuthorizationResourceTypes,
        resource_name: str,
        action: schemas.AuthorizationAction,
        auth_info: schemas.AuthInfo,
        raise_on_forbidden: bool = True,
        resource_namespace: schemas.AuthorizationResourceNamespace = schemas.AuthorizationResourceNamespace.resources,
    ) -> bool:
        return await self.query_permissions(
            self._attach_resource_namespace(
                resource_type.to_resource_string("", resource_name),
                resource_namespace,
            ),
            action=action,
            auth_info=auth_info,
            raise_on_forbidden=raise_on_forbidden,
        )

    async def query_permissions(
        self,
        resource: str,
        action: schemas.AuthorizationAction,
        auth_info: schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        return await self._auth_provider.query_permissions(
            resource, action, auth_info, raise_on_forbidden
        )

    async def filter_by_permissions(
        self,
        resources: list,
        opa_resource_extractor: typing.Callable,
        action: schemas.AuthorizationAction,
        auth_info: schemas.AuthInfo,
    ) -> list:
        return await self._auth_provider.filter_by_permissions(
            resources,
            opa_resource_extractor,
            action,
            auth_info,
        )

    def add_allowed_project_for_owner(
        self, project_name: str, auth_info: schemas.AuthInfo
    ):
        self._auth_provider.add_allowed_project_for_owner(project_name, auth_info)

    async def authenticate_request(self, request: fastapi.Request) -> schemas.AuthInfo:
        auth_info = schemas.AuthInfo()
        headers = request.headers

        if self._basic_auth_configured():
            auth_info = self._authenticate_basic(headers)
        elif self._bearer_auth_configured():
            auth_info = self._authenticate_bearer(headers)
        elif self._iguazio_auth_configured():
            auth_info = await self._authenticate_iguazio(request)
        elif self._iguaziov4_auth_configured():
            auth_info = await self._authenticate_iguazio_v4(request)

        # Fallback in case auth method didn't fill in the username already, and it is provided by the caller
        if not auth_info.username and schemas.HeaderNames.remote_user in headers:
            auth_info.username = headers[schemas.HeaderNames.remote_user]

        projects_role_header = headers.get(schemas.HeaderNames.projects_role)
        auth_info.projects_role = (
            schemas.ProjectsRole(projects_role_header) if projects_role_header else None
        )
        # In Iguazio 3.0 we're running with auth mode none cause auth is done by the ingress, in that auth mode sessions
        # needed for data operations were passed through this header, keep reading it to be backwards compatible
        if (
            not auth_info.data_session
            and schemas.HeaderNames.v3io_session_key in headers
        ):
            auth_info.data_session = headers[schemas.HeaderNames.v3io_session_key]
        # In Iguazio 3.0 the ingress auth verification overrides the X-V3io-Session-Key from the auth response
        # therefore the above won't work for requests coming from outside the cluster so allowing another header that
        # won't be overridden
        if (
            not auth_info.data_session
            and schemas.HeaderNames.v3io_access_key in headers
        ):
            auth_info.data_session = headers[schemas.HeaderNames.v3io_access_key]

        # Maintain authentication headers for inter-services communication
        auth_info.request_headers = dict(headers)
        for header in [
            "content-length",
            "content-type",
        ]:
            auth_info.request_headers.pop(header, None)

        # mask clients host with worker's host
        origin_host = auth_info.request_headers.pop("host", None)
        if origin_host:
            # original host requested by client
            auth_info.request_headers[schemas.HeaderNames.forwarded_host] = origin_host
        return auth_info

    def get_or_create_access_key(
        self, session: str, planes: list[str] | None = None
    ) -> str:
        if not self._iguazio_auth_configured():
            raise NotImplementedError(
                "Access key is currently supported only for Iguazio authentication mode"
            )
        return framework.utils.clients.iguazio.v3.Client().get_or_create_access_key(
            session, planes
        )

    def is_jobs_auth_required(self):
        return self._iguazio_auth_configured()

    def _generate_resource_string_from_project_name(
        self,
        project_name: str,
        resource_namespace: schemas.AuthorizationResourceNamespace,
    ):
        return self._attach_resource_namespace(
            schemas.AuthorizationResourceTypes.project.to_resource_string(
                project_name, ""
            ),
            resource_namespace,
        )

    async def ensure_project_permissions(
        self,
        project_name: str,
        auth_info: mlrun.common.schemas.AuthInfo,
    ):
        """
        Ensures project permissions are populated in the AuthVerifier
        """

        async def _check_project_read_permissions():
            await self.query_project_permissions(
                project_name,
                mlrun.common.schemas.AuthorizationAction.read,
                auth_info,
            )

        await mlrun.utils.helpers.retry_until_successful_async(
            backoff=1,
            timeout=10,
            logger=logger,
            verbose=False,
            _function=_check_project_read_permissions,
        )

    def _generate_resource_string_from_project_resource(
        self,
        resource_type: schemas.AuthorizationResourceTypes,
        project_name: str,
        resource_name: str,
        resource_namespace: schemas.AuthorizationResourceNamespace,
    ):
        if not project_name:
            project_name = "*"
        if not resource_name:
            resource_name = "*"
        return self._attach_resource_namespace(
            resource_type.to_resource_string(project_name, resource_name),
            resource_namespace,
        )

    def _attach_resource_namespace(
        self,
        resource: str,
        resource_namespace: schemas.AuthorizationResourceNamespace,
    ) -> str:
        if namespace := self._prefixes[resource_namespace]:
            return f"/{namespace}{resource}"
        return resource

    @staticmethod
    def _basic_auth_configured():
        return mlrun.mlconf.httpdb.authentication.mode == AuthenticationMode.BASIC and (
            mlrun.mlconf.httpdb.authentication.basic.username
            or mlrun.mlconf.httpdb.authentication.basic.password
        )

    @staticmethod
    def _bearer_auth_configured():
        return (
            mlrun.mlconf.httpdb.authentication.mode == AuthenticationMode.BEARER
            and mlrun.mlconf.httpdb.authentication.bearer.token
        )

    @staticmethod
    def _iguazio_auth_configured():
        return mlrun.mlconf.is_iguazio_mode()

    @staticmethod
    def _iguaziov4_auth_configured():
        return mlrun.mlconf.is_iguazio_v4_mode()

    @staticmethod
    def _parse_basic_auth(header):
        """
        parse_basic_auth('Basic YnVnczpidW5ueQ==')
        ['bugs', 'bunny']
        """
        b64value = header[len(schemas.AuthorizationHeaderPrefixes.basic) :]
        value = base64.b64decode(b64value).decode()
        return value.split(":", 1)

    def _authenticate_basic(
        self, headers: typing.Mapping[str, str]
    ) -> schemas.AuthInfo:
        header = headers.get(schemas.HeaderNames.authorization, "")
        if not header.startswith(schemas.AuthorizationHeaderPrefixes.basic):
            raise mlrun.errors.MLRunUnauthorizedError("Missing basic auth header")

        username, password = self._parse_basic_auth(header)
        if (
            username != mlrun.mlconf.httpdb.authentication.basic.username
            or password != mlrun.mlconf.httpdb.authentication.basic.password
        ):
            raise mlrun.errors.MLRunUnauthorizedError(
                "Username or password did not match"
            )

        return schemas.AuthInfo(username=username, password=password)

    def _authenticate_bearer(
        self, headers: typing.Mapping[str, str]
    ) -> schemas.AuthInfo:
        header = headers.get(schemas.HeaderNames.authorization, "")
        if not header.startswith(schemas.AuthorizationHeaderPrefixes.bearer):
            raise mlrun.errors.MLRunUnauthorizedError("Missing bearer auth header")

        token = header[len(schemas.AuthorizationHeaderPrefixes.bearer) :]
        if token != mlrun.mlconf.httpdb.authentication.bearer.token:
            raise mlrun.errors.MLRunUnauthorizedError("Token did not match")

        return schemas.AuthInfo(token=token)

    @staticmethod
    async def _authenticate_iguazio(
        request: fastapi.Request,
    ) -> schemas.AuthInfo:
        iguazio_client = framework.utils.clients.iguazio.v3.AsyncClient()
        auth_info = await iguazio_client.verify_request_session(request)
        if schemas.HeaderNames.data_session_override in request.headers:
            auth_info.data_session = request.headers[
                schemas.HeaderNames.data_session_override
            ]
        return auth_info

    async def _authenticate_iguazio_v4(
        self,
        request: fastapi.Request,
    ) -> schemas.AuthInfo:
        token = self._extract_token(request)
        curr_time = time.time()

        if token is None or token[1] <= curr_time:
            # No token or an expired token means no caching
            # TODO: should we immediately throw an error instead of trying to
            # verify it?
            iguazio_client = framework.utils.clients.iguazio.v4.AsyncClient()
            return await iguazio_client.verify_request_session(request)

        token, expires_at = token

        self._expire_tokens(curr_time)
        task = self._token_cache.get(token)

        if task is None:
            # No task means we have to create it and add it to the cache
            iguazio_client = framework.utils.clients.iguazio.v4.AsyncClient()
            task = asyncio.create_task(iguazio_client.verify_request_session(request))
            task.add_done_callback(partial(self._on_verify_complete, token))

            self._token_cache[token] = task
            if len(self._token_cache) > TOKEN_CACHE_MAX_SIZE:
                self._token_cache.popitem(last=False)

            expires_at = min(expires_at, curr_time + TOKEN_CACHE_MAX_TTL)
            heappush(self._token_expiry_heap, TokenExpiryEntry(token, task, expires_at))
        else:
            # Move to end to mark it as the most recently used
            self._token_cache.move_to_end(token)

        # We shield the task since it can be shared between multiple
        # verifications and cancellation could have unexpected side effects
        return await asyncio.shield(task)

    @staticmethod
    def _extract_token(request: fastapi.Request) -> tuple[str, float] | None:
        header_value = request.headers.get("Authorization")
        if header_value is None:
            return None

        scheme, _, token = header_value.partition(" ")
        if scheme.lower() != "bearer":  # scheme is case insensitive
            return None

        # We don't verify the signature here, this is handled by the endpoint
        # called by the iguazio_client, we just want to extract the expiry time
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
        except jwt.DecodeError:
            return None

        expires_at = decoded.get("exp")
        if not isinstance(expires_at, (int, float)):
            return None

        return token, expires_at

    def _expire_tokens(self, curr_time: float) -> None:
        # The heap can have items that have been removed already, to keep this
        # in bounds with the cache size we clean up the entire heap if it is
        # more than double the cache size
        if len(self._token_expiry_heap) > len(self._token_cache) * 2:
            # In this case we filter the whole list while handling expiry and
            # heapify it again
            new_len = 0

            for entry in self._token_expiry_heap:
                if self._token_cache.get(entry.token) is not entry.task:
                    pass  # already evicted
                elif entry.expires_at <= curr_time:
                    del self._token_cache[entry.token]  # expired
                else:
                    self._token_expiry_heap[new_len] = entry
                    new_len += 1

            del self._token_expiry_heap[new_len:]
            heapify(self._token_expiry_heap)
        else:
            # We pop expired tokens from the heap to evict them from the cache
            while (
                self._token_expiry_heap
                and self._token_expiry_heap[0].expires_at <= curr_time
            ):
                entry = heappop(self._token_expiry_heap)
                if self._token_cache.get(entry.token) is entry.task:
                    del self._token_cache[entry.token]

    def _on_verify_complete(
        self, token: str, task: asyncio.Task[schemas.AuthInfo]
    ) -> None:
        # We evict from the cache on failure to make sure we dont block tokens
        # on things like temporary connectivity issues
        # TODO: should we whitelist certain exceptions to be kept? e.g. invalid token
        task_failed = task.cancelled() or task.exception() is not None
        if task_failed and self._token_cache.get(token) is task:
            del self._token_cache[token]
