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
import contextlib
import copy
import datetime
import typing

import humanfriendly

import mlrun.api.schemas
import mlrun.api.utils.auth.providers.base
import mlrun.api.utils.projects.remotes.leader
import mlrun.errors
import mlrun.utils.helpers
import mlrun.utils.singleton
from mlrun.utils import logger


class Provider(
    mlrun.api.utils.auth.providers.base.Provider,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self) -> None:
        super().__init__()
        self._session: typing.Optional[mlrun.utils.AsyncClientWithRetry] = None
        self._api_url = mlrun.mlconf.httpdb.authorization.opa.address
        self._permission_query_path = (
            mlrun.mlconf.httpdb.authorization.opa.permission_query_path
        )
        self._permission_filter_path = (
            mlrun.mlconf.httpdb.authorization.opa.permission_filter_path
            # a little hack to make this work until the provisioners of mlrun start configuring the filter path
            # TODO: remove me
            or mlrun.mlconf.httpdb.authorization.opa.permission_query_path.replace(
                "allow", "filter_allowed"
            )
        )
        self._request_timeout = int(
            mlrun.mlconf.httpdb.authorization.opa.request_timeout
        )
        self._log_level = int(mlrun.mlconf.httpdb.authorization.opa.log_level)
        self._leader_name = mlrun.mlconf.httpdb.projects.leader
        self._allowed_project_owners_cache_ttl_seconds = humanfriendly.parse_timespan(
            mlrun.mlconf.httpdb.projects.project_owners_cache_ttl
        )

        # owner id -> allowed project -> ttl
        self._allowed_project_owners_cache: typing.Dict[
            str, typing.Dict[str, datetime]
        ] = {}

    async def query_permissions(
        self,
        resource: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:

        # store is not really a verb in our OPA manifest, we map it to 2 query permissions requests (create & update)
        if action == mlrun.api.schemas.AuthorizationAction.store:
            results = await asyncio.gather(
                self.query_permissions(
                    resource,
                    mlrun.api.schemas.AuthorizationAction.create,
                    auth_info,
                    raise_on_forbidden,
                ),
                self.query_permissions(
                    resource,
                    mlrun.api.schemas.AuthorizationAction.update,
                    auth_info,
                    raise_on_forbidden,
                ),
            )
            create_allowed, update_allowed = results
            return create_allowed and update_allowed
        if self._is_request_from_leader(auth_info.projects_role):
            return True
        if self._check_allowed_project_owners_cache(resource, auth_info):
            return True
        body = self._generate_permission_request_body(resource, action, auth_info)
        if self._log_level > 5:
            logger.debug("Sending request to OPA", body=body)
        async with self._send_request_to_api(
            "POST", self._permission_query_path, json=body
        ) as response:
            response_body = await response.json()
        if self._log_level > 5:
            logger.debug("Received response from OPA", body=response_body)
        allowed = response_body["result"]
        if not allowed and raise_on_forbidden:
            raise mlrun.errors.MLRunAccessDeniedError(
                f"Not allowed to {action} resource {resource}"
            )
        return allowed

    async def filter_by_permissions(
        self,
        resources: typing.List,
        opa_resource_extractor: typing.Callable,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
    ) -> typing.List:
        # store is not really a verb in our OPA manifest, we map it to 2 query permissions requests (create & update)
        if action == mlrun.api.schemas.AuthorizationAction.store:
            raise NotImplementedError("Store action is not supported in filtering")
        if self._is_request_from_leader(auth_info.projects_role):
            return resources
        opa_resources = []
        for resource in resources:
            opa_resources.append(opa_resource_extractor(resource))
        allowed_by_cache = True
        for opa_resource in opa_resources:
            # allow by cache only if all resources allowed by cache
            if not self._check_allowed_project_owners_cache(opa_resource, auth_info):
                allowed_by_cache = False
                break
        if allowed_by_cache:
            return resources
        body = self._generate_filter_request_body(opa_resources, action, auth_info)
        if self._log_level > 5:
            logger.debug("Sending filter request to OPA", body=body)
        async with self._send_request_to_api(
            "POST", self._permission_filter_path, json=body
        ) as response:
            response_body = await response.json()
        if self._log_level > 5:
            logger.debug("Received filter response from OPA", body=response_body)
        allowed_opa_resources = response_body["result"]
        allowed_resources = []
        for index, opa_resource in enumerate(opa_resources):
            if opa_resource in allowed_opa_resources:
                allowed_resources.append(resources[index])
        return allowed_resources

    def add_allowed_project_for_owner(
        self, project_name: str, auth_info: mlrun.api.schemas.AuthInfo
    ):
        if (
            not auth_info.user_id
            or not project_name
            or not self._allowed_project_owners_cache_ttl_seconds
        ):
            # Simply won't be cached, no need to explode
            return
        allowed_projects = {}
        if auth_info.user_id in self._allowed_project_owners_cache:
            allowed_projects = self._allowed_project_owners_cache[auth_info.user_id]
        ttl = datetime.datetime.now() + datetime.timedelta(
            seconds=self._allowed_project_owners_cache_ttl_seconds
        )
        allowed_projects[project_name] = ttl
        self._allowed_project_owners_cache[auth_info.user_id] = allowed_projects

    def _check_allowed_project_owners_cache(
        self, resource: str, auth_info: mlrun.api.schemas.AuthInfo
    ):
        # Cache shouldn't be big, simply clean it on get instead of scheduling it
        self._clean_expired_records_from_cache()
        if auth_info.user_id not in self._allowed_project_owners_cache:
            return False
        allowed_projects = list(
            self._allowed_project_owners_cache[auth_info.user_id].keys()
        )
        for allowed_project in allowed_projects:
            if f"/projects/{allowed_project}/" in resource:
                return True
        return False

    def _clean_expired_records_from_cache(self):
        now = datetime.datetime.now()
        user_ids_to_remove = []
        for user_id in self._allowed_project_owners_cache.keys():
            allowed_projects = self._allowed_project_owners_cache[user_id]
            updated_allowed_projects = {}
            for allowed_project_name, ttl in allowed_projects.items():
                if now > ttl:
                    continue
                updated_allowed_projects[allowed_project_name] = ttl
            self._allowed_project_owners_cache[user_id] = updated_allowed_projects
            if not updated_allowed_projects:
                user_ids_to_remove.append(user_id)
        for user_id in user_ids_to_remove:
            del self._allowed_project_owners_cache[user_id]

    def _is_request_from_leader(
        self, projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole]
    ):
        if projects_role and projects_role.value == self._leader_name:
            return True
        return False

    @contextlib.asynccontextmanager
    async def _send_request_to_api(self, method, path, **kwargs):
        url = f"{self._api_url}{path}"
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self._request_timeout
        await self._ensure_session()
        response = None
        try:
            response = await self._session.request(
                method, url, verify_ssl=False, **kwargs
            )
            if not response.ok:
                await self._on_request_api_failure(method, path, response, **kwargs)
            yield response
        finally:
            if response:
                response.release()

    async def _on_request_api_failure(self, method, path, response, **kwargs):
        log_kwargs = copy.deepcopy(kwargs)
        log_kwargs.update({"method": method, "path": path})
        if response.content:
            try:
                data = await response.json()
            except Exception:
                try:
                    data = await response.text()
                except Exception:
                    data = None
            log_kwargs.update({"data": data})
        logger.warning("Request to opa failed", **log_kwargs)
        mlrun.errors.raise_for_status(response)

    @staticmethod
    def _generate_permission_request_body(
        resource: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
    ) -> dict:
        body = {
            "input": {
                "resource": resource,
                "action": str(action),
                "ids": auth_info.get_member_ids(),
            },
        }
        return body

    @staticmethod
    def _generate_filter_request_body(
        resources: typing.List[str],
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
    ) -> dict:
        body = {
            "input": {
                "resources": resources,
                "action": str(action),
                "ids": auth_info.get_member_ids(),
            },
        }
        return body

    async def _ensure_session(self):
        if not self._session:
            self._session = mlrun.utils.AsyncClientWithRetry(
                logger=logger,
            )
