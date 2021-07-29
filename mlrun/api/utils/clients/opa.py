import copy
import datetime
import typing

import humanfriendly
import requests.adapters
import urllib3

import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.leader
import mlrun.errors
import mlrun.utils.helpers
import mlrun.utils.singleton
from mlrun.utils import logger


class Client(metaclass=mlrun.utils.singleton.Singleton,):
    def __init__(self) -> None:
        super().__init__()
        http_adapter = requests.adapters.HTTPAdapter(
            max_retries=urllib3.util.retry.Retry(total=3, backoff_factor=1)
        )
        self._session = requests.Session()
        self._session.mount("http://", http_adapter)
        self._api_url = mlrun.mlconf.httpdb.authorization.opa.address
        self._permission_query_path = (
            mlrun.mlconf.httpdb.authorization.opa.permission_query_path
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

    def filter_resources_by_permissions(
        self,
        resource_type: mlrun.api.schemas.AuthorizationResourceTypes,
        resources: typing.List,
        project_and_resource_name_extractor: typing.Callable,
        auth_info: mlrun.api.schemas.AuthInfo,
    ) -> typing.List:
        # TODO: execute in parallel
        filtered_resources = []
        for resource in resources:
            project_name, resource_name = project_and_resource_name_extractor(resource)
            allowed = self.query_resource_permissions(
                resource_type,
                project_name,
                resource_name,
                mlrun.api.schemas.AuthorizationAction.read,
                auth_info,
                raise_on_forbidden=False,
            )
            if allowed:
                filtered_resources.append(resource)
        return filtered_resources

    def query_resources_permissions(
        self,
        resource_type: mlrun.api.schemas.AuthorizationResourceTypes,
        resources: typing.List,
        project_and_resource_name_extractor: typing.Callable,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        allowed = True
        # TODO: execute in parallel
        for resource in resources:
            project_name, resource_name = project_and_resource_name_extractor(resource)
            resource_allowed = self.query_resource_permissions(
                resource_type,
                project_name,
                resource_name,
                action,
                auth_info,
                raise_on_forbidden,
            )
            allowed = allowed and resource_allowed
        return allowed

    def query_resource_permissions(
        self,
        resource_type: mlrun.api.schemas.AuthorizationResourceTypes,
        project_name: str,
        resource_name: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        if not project_name:
            project_name = "*"
        if not resource_name:
            resource_name = "*"
        return self.query_permissions(
            resource_type.to_resource_string(project_name, resource_name),
            action,
            auth_info,
            raise_on_forbidden,
        )

    def query_permissions(
        self,
        resource: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        # store is not really a verb in our OPA manifest, we map it to 2 query permissions requests (create & update)
        if action == mlrun.api.schemas.AuthorizationAction.store:
            create_allowed = self.query_permissions(
                resource,
                mlrun.api.schemas.AuthorizationAction.create,
                auth_info,
                raise_on_forbidden,
            )
            update_allowed = self.query_permissions(
                resource,
                mlrun.api.schemas.AuthorizationAction.update,
                auth_info,
                raise_on_forbidden,
            )
            return create_allowed and update_allowed
        if (
            self._is_request_from_leader(auth_info.projects_role)
            or mlrun.mlconf.httpdb.authorization.mode == "none"
        ):
            return True
        if self._check_allowed_project_owners_cache(resource, auth_info):
            return True
        body = self._generate_permission_request_body(resource, action, auth_info)
        if self._log_level > 5:
            logger.debug("Sending request to OPA", body=body)
        response = self._send_request_to_api(
            "POST", self._permission_query_path, json=body
        )
        response_body = response.json()
        if self._log_level > 5:
            logger.debug("Received response from OPA", body=response_body)
        allowed = response_body["result"]
        if not allowed and raise_on_forbidden:
            raise mlrun.errors.MLRunAccessDeniedError(
                f"Not allowed to {action} resource {resource}"
            )
        return allowed

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

    def _is_request_from_leader(
        self, projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole]
    ):
        if projects_role and projects_role.value == self._leader_name:
            return True
        return False

    def _send_request_to_api(self, method, path, **kwargs):
        url = f"{self._api_url}{path}"
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self._request_timeout
        response = self._session.request(method, url, verify=False, **kwargs)
        if not response.ok:
            log_kwargs = copy.deepcopy(kwargs)
            log_kwargs.update({"method": method, "path": path})
            if response.content:
                try:
                    data = response.json()
                except Exception:
                    pass
                else:
                    log_kwargs.update({"data": data})
            logger.warning("Request to opa failed", **log_kwargs)
            mlrun.errors.raise_for_status(response)
        return response

    @staticmethod
    def _generate_permission_request_body(
        resource: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
    ) -> dict:
        body = {
            "input": {
                "resource": resource,
                "action": action,
                "ids": auth_info.get_member_ids(),
            },
        }
        return body
