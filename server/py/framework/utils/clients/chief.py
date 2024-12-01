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
import contextlib
import typing

import aiohttp
import fastapi

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton

import framework.utils.clients.messaging


# we were thinking to simply use httpdb, but decided to have a separated class for simplicity for now until
# this class evolves, but this should be reconsidered when adding more functionality to the class
class Client(
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    """
    We chose chief-workers architecture to provide multi-instance API.
    By default, all API calls can access both the chief and workers.
    The key distinction is that some responsibilities, such as scheduling jobs, are exclusively performed by the chief.
    Instead of limiting the ui/client to only send requests to the chief, because the workers doesn't hold all the
    information. When one of the workers receives a request that the chief needs to execute or may have the knowledge
    of that piece of information, the worker will redirect the request to the chief.
    """

    def __init__(self) -> None:
        super().__init__()
        self._messaging_client = framework.utils.clients.messaging.Client()
        self._api_url = mlrun.mlconf.resolve_chief_api_url()
        self._api_url = self._api_url.rstrip("/")

    async def get_internal_background_task(
        self, name: str, request: fastapi.Request = None
    ) -> fastapi.Response:
        """
        internal background tasks are managed by the chief only
        """
        return await self._proxy_request_to_chief(
            "GET", f"background-tasks/{name}", request
        )

    async def get_internal_background_tasks(
        self, request: fastapi.Request = None
    ) -> fastapi.Response:
        """
        internal background tasks are managed by the chief only
        """
        return await self._proxy_request_to_chief("GET", "background-tasks", request)

    async def trigger_migrations(
        self, request: fastapi.Request = None
    ) -> fastapi.Response:
        """
        only chief can execute migrations
        """
        return await self._proxy_request_to_chief(
            "POST", "operations/migrations", request
        )

    async def refresh_smtp_configuration(
        self, request: fastapi.Request = None
    ) -> fastapi.Response:
        """
        To avoid raise condition we want only that the chief will store the secret in the k8s secret store
        """
        return await self._proxy_request_to_chief(
            "POST", "operations/refresh-smtp-configuration", request
        )

    async def create_schedule(
        self, project: str, request: fastapi.Request, json: dict
    ) -> fastapi.Response:
        """
        Schedules are running only on chief
        """
        return await self._proxy_request_to_chief(
            "POST", f"projects/{project}/schedules", request, json
        )

    async def update_schedule(
        self, project: str, name: str, request: fastapi.Request, json: dict
    ) -> fastapi.Response:
        """
        Schedules are running only on chief
        """
        return await self._proxy_request_to_chief(
            "PUT", f"projects/{project}/schedules/{name}", request, json
        )

    async def delete_schedule(
        self, project: str, name: str, request: fastapi.Request
    ) -> fastapi.Response:
        """
        Schedules are running only on chief
        """
        return await self._proxy_request_to_chief(
            "DELETE", f"projects/{project}/schedules/{name}", request
        )

    async def submit_workflow(
        self,
        project: str,
        name: str,
        request: fastapi.Request,
        json: dict,
    ) -> fastapi.Response:
        """
        Workflow schedules are running only on chief
        """
        return await self._proxy_request_to_chief(
            "POST", f"projects/{project}/workflows/{name}/submit", request, json
        )

    async def delete_schedules(
        self, project: str, request: fastapi.Request
    ) -> fastapi.Response:
        """
        Schedules are running only on chief
        """
        return await self._proxy_request_to_chief(
            "DELETE", f"projects/{project}/schedules", request
        )

    async def invoke_schedule(
        self, project: str, name: str, request: fastapi.Request
    ) -> fastapi.Response:
        """
        Schedules are running only on chief
        """
        return await self._proxy_request_to_chief(
            "POST", f"projects/{project}/schedules/{name}/invoke", request
        )

    async def submit_job(
        self, request: fastapi.Request, json: dict
    ) -> fastapi.Response:
        """
        submit job can be responsible for creating schedules and schedules are running only on chief,
        so when the job contains a schedule, we re-route the request to chief
        """
        return await self._proxy_request_to_chief(
            "POST",
            "submit_job",
            request,
            json,
            timeout=int(mlrun.mlconf.submit_timeout),
        )

    async def build_function(
        self, request: fastapi.Request, json: dict
    ) -> fastapi.Response:
        """
        if serving function and track_models is enabled, it means that schedules will be created as part of
        building the function, then we re-route the request to chief
        """
        return await self._proxy_request_to_chief(
            "POST", "build/function", request, json
        )

    async def delete_project(
        self, name, request: fastapi.Request, api_version: typing.Optional[str] = None
    ) -> fastapi.Response:
        """
        delete project can be responsible for deleting schedules. Schedules are running only on chief,
        that is why we re-route requests to chief
        """
        # timeout is greater than default as delete project can take a while because it deletes all the
        # project resources (depends on the deletion strategy and api version)
        return await self._proxy_request_to_chief(
            "DELETE", f"projects/{name}", request, timeout=120, version=api_version
        )

    async def get_clusterization_spec(
        self, return_fastapi_response: bool = True, raise_on_failure: bool = False
    ) -> typing.Union[fastapi.Response, mlrun.common.schemas.ClusterizationSpec]:
        """
        This method is used both for proxying requests from worker to chief and for aligning the worker state
        with the clusterization spec brought from the chief
        """
        async with self._send_request_to_api(
            method="GET",
            path="clusterization-spec",
            raise_on_failure=raise_on_failure,
        ) as chief_response:
            if return_fastapi_response:
                return await self._messaging_client.convert_requests_response_to_fastapi_response(
                    chief_response
                )

            return mlrun.common.schemas.ClusterizationSpec(
                **(await chief_response.json())
            )

    async def store_alert_template(
        self, name: str, request: fastapi.Request, json: dict
    ) -> fastapi.Response:
        """
        Alert templates are running only on chief
        """
        return await self._proxy_request_to_chief(
            "PUT", f"alert_templates/{name}", request, json
        )

    async def delete_alert_template(
        self, name: str, request: fastapi.Request
    ) -> fastapi.Response:
        """
        Alert templates are running only on chief
        """
        return await self._proxy_request_to_chief(
            "DELETE", f"alert_templates/{name}", request
        )

    async def store_alert(
        self, project: str, name: str, request: fastapi.Request, json: dict
    ) -> typing.Union[fastapi.Response, mlrun.common.schemas.AlertConfig]:
        """
        Alerts are running only on chief
        """
        return await self._proxy_request_to_chief(
            "PUT", f"projects/{project}/alerts/{name}", request, json
        )

    async def delete_alert(
        self, project: str, name: str, request: fastapi.Request
    ) -> fastapi.Response:
        """
        Alerts are running only on chief
        """
        return await self._proxy_request_to_chief(
            "DELETE", f"projects/{project}/alerts/{name}", request
        )

    async def reset_alert(
        self, project: str, name: str, request: fastapi.Request
    ) -> fastapi.Response:
        """
        Alerts are running only on chief
        """
        return await self._proxy_request_to_chief(
            "POST", f"projects/{project}/alerts/{name}/reset", request
        )

    async def set_event(
        self, project: str, name: str, request: fastapi.Request, json: dict
    ) -> fastapi.Response:
        """
        Events are running only on chief
        """
        return await self._proxy_request_to_chief(
            "POST", f"projects/{project}/events/{name}", request, json
        )

    async def set_schedule_notifications(
        self, project: str, schedule_name: str, request: fastapi.Request, json: dict
    ) -> fastapi.Response:
        """
        Schedules are running only on chief
        """
        return await self._proxy_request_to_chief(
            "PUT",
            f"projects/{project}/schedules/{schedule_name}/notifications",
            request,
            json,
        )

    async def _proxy_request_to_chief(
        self,
        method,
        path,
        request: fastapi.Request = None,
        json: typing.Optional[dict] = None,
        version: typing.Optional[str] = None,
        raise_on_failure: bool = False,
        **kwargs,
    ) -> fastapi.Response:
        service_name, url = self._resolve_chief_params(path, version)
        return await self._messaging_client.proxy_request_to_service(
            service_name=service_name,
            method=method,
            url=url,
            request=request,
            json=json,
            raise_on_failure=raise_on_failure,
            **kwargs,
        )

    @contextlib.asynccontextmanager
    async def _send_request_to_api(
        self,
        method,
        path,
        version: typing.Optional[str] = None,
        raise_on_failure: bool = False,
        **kwargs,
    ) -> aiohttp.ClientResponse:
        service_name, url = self._resolve_chief_params(path, version)
        async with self._messaging_client.send_request(
            service_name=service_name,
            method=method,
            url=url,
            raise_on_failure=raise_on_failure,
            **kwargs,
        ) as response:
            yield response

    def _resolve_chief_params(self, path, version):
        service_name = "api-chief"
        version = version or mlrun.mlconf.api_base_version
        url = f"{self._api_url}/api/{version}/{path}"
        return service_name, url
