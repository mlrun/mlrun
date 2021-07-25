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
            self._generate_resource_string(project_name, resource_type, resource_name),
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
        self,
        project_name: str,
        resource_type: mlrun.api.schemas.AuthorizationResourceTypes,
        resource_name: str,
    ):
        return {
            mlrun.api.schemas.AuthorizationResourceTypes.function: "/projects/{project_name}/functions/{resource_name}",
            mlrun.api.schemas.AuthorizationResourceTypes.artifact: "/projects/{project_name}/artifacts/{resource_name}",
            mlrun.api.schemas.AuthorizationResourceTypes.background_task: "/projects/{project_name}/background-tasks/{r"
            "esource_name}",
            mlrun.api.schemas.AuthorizationResourceTypes.feature_set: "/projects/{project_name}/feature-sets/{resource_"
            "name}",
            mlrun.api.schemas.AuthorizationResourceTypes.feature_vector: "/projects/{project_name}/feature-vectors/{res"
            "ource_name}",
            mlrun.api.schemas.AuthorizationResourceTypes.feature: "/projects/{project_name}/features/{resource_name}",
            mlrun.api.schemas.AuthorizationResourceTypes.entity: "/projects/{project_name}/entities/{resource_name}",
            mlrun.api.schemas.AuthorizationResourceTypes.log: "/projects/{project_name}/runs/{resource_name}/logs",
            mlrun.api.schemas.AuthorizationResourceTypes.schedule: "/projects/{project_name}/schedules/{resource_name}",
            mlrun.api.schemas.AuthorizationResourceTypes.secret: "/projects/{project_name}/secrets/{resource_name}",
            mlrun.api.schemas.AuthorizationResourceTypes.run: "/projects/{project_name}/runs/{resource_name}",
            # runtime resource doesn't have a get (one) object endpoint, it doesn't have an identifier
            mlrun.api.schemas.AuthorizationResourceTypes.runtime_resource: "/projects/{project_name}/runtime-resources/"
            "",
            mlrun.api.schemas.AuthorizationResourceTypes.model_endpoint: "/projects/{project_name}/model-endpoints/{res"
            "ource_name}",
        }[resource_type].format(project_name=project_name, resource_name=resource_name)

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
