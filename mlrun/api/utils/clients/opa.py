import copy
import typing

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

    def filter_projects_by_permissions(
        self, project_names: typing.List[str], auth_info: mlrun.api.schemas.AuthInfo,
    ) -> typing.List[str]:
        # TODO: execute in parallel
        filtered_projects = []
        for project_name in project_names:
            allowed = self.query_project_permissions(
                project_name,
                mlrun.api.schemas.AuthorizationAction.read,
                auth_info,
                raise_on_forbidden=False,
            )
            if allowed:
                filtered_projects.append(project_name)
        return filtered_projects

    def query_project_permissions(
        self,
        project_name: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        if not project_name:
            project_name = "*"
        return self.query_permissions(
            self._generate_resource_string(project_name, "project", project_name),
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
        if (
            self._is_request_from_leader(auth_info.projects_role)
            or mlrun.mlconf.httpdb.authorization.mode == "none"
        ):
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

    def _generate_resource_string(
        self, project_name: str, resource_type: str, resource: str
    ):
        return {
            "project": "/projects/{project_name}",
            "function": "/projects/{project_name}/functions/{resource}",
        }[resource_type].format(project_name=project_name, resource=resource)

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
        member_ids = []
        if auth_info.user_id:
            member_ids.append(auth_info.user_id)
        if auth_info.user_group_ids:
            member_ids.extend(auth_info.user_group_ids)
        body = {
            "input": {"resource": resource, "action": action, "ids": member_ids},
        }
        return body
