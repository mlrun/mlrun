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
import copy
import json
import urllib.parse

import aiohttp
from http import HTTPStatus

import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes.nuclio.api_gateway
import mlrun.utils
from mlrun.common.constants import MLRUN_CREATED_LABEL
from mlrun.utils import logger

NUCLIO_API_SESSIONS_ENDPOINT = "/api/sessions/"
NUCLIO_API_GATEWAYS_ENDPOINT_TEMPLATE = "/api/api_gateways/{api_gateway}"
NUCLIO_API_GATEWAY_NAMESPACE_HEADER = "X-Nuclio-Api-Gateway-Namespace"
NUCLIO_PROJECT_NAME_HEADER = "X-Nuclio-Project-Name"
NUCLIO_PROJECT_NAME_LABEL = "nuclio.io/project-name"


class Client:
    def __init__(self, auth_info: mlrun.common.schemas.AuthInfo):
        self._session = None
        self._auth = aiohttp.BasicAuth(auth_info.username, auth_info.session)
        self._logger = logger.get_child("nuclio-client")
        self._nuclio_dashboard_url = mlrun.mlconf.nuclio_dashboard_url
        self._nuclio_domain = urllib.parse.urlparse(self._nuclio_dashboard_url).netloc

    async def __aenter__(self):
        await self._ensure_async_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._close_session()

    async def list_api_gateways(
        self, project_name=None
    ) -> dict[str, mlrun.common.schemas.APIGateway]:
        headers = {}

        if project_name:
            headers[NUCLIO_PROJECT_NAME_HEADER] = project_name

        api_gateways = await self._send_request_to_api(
            method="GET",
            path=NUCLIO_API_GATEWAYS_ENDPOINT_TEMPLATE.format(api_gateway=""),
            headers=headers,
        )
        parsed_api_gateways = {}
        for name, gw in api_gateways.items():
            parsed_api_gateways[name] = mlrun.common.schemas.APIGateway.parse_obj(gw)
        return parsed_api_gateways

    async def api_gateway_exists(self, name: str, project_name: str = None):
        return name in await self.list_api_gateways(project_name=project_name)

    async def get_api_gateway(self, name: str, project_name: str = None):
        headers = {}

        if project_name:
            headers[NUCLIO_PROJECT_NAME_HEADER] = project_name

        api_gateway = await self._send_request_to_api(
            method="GET",
            path=NUCLIO_API_GATEWAYS_ENDPOINT_TEMPLATE.format(api_gateway=name),
            headers=headers,
        )
        return mlrun.common.schemas.APIGateway.parse_obj(api_gateway)

    async def store_api_gateway(
        self,
        project_name: str,
        api_gateway_name: str,
        api_gateway: mlrun.common.schemas.APIGateway,
        create: bool = False,
    ):
        headers = {}
        self._enrich_nuclio_api_gateway(
            project_name=project_name,
            api_gateway=api_gateway,
            api_gateway_name=api_gateway_name,
        )

        if project_name:
            headers[NUCLIO_PROJECT_NAME_HEADER] = project_name

        body = api_gateway.dict(exclude_unset=True, exclude_none=True)
        method = "POST" if create else "PUT"
        path = (
            NUCLIO_API_GATEWAYS_ENDPOINT_TEMPLATE.format(api_gateway=api_gateway_name)
            if method == "PUT"
            else NUCLIO_API_GATEWAYS_ENDPOINT_TEMPLATE.format(api_gateway="")
        )

        return await self._send_request_to_api(
            method=method,
            path=path,
            headers=headers,
            json=body,
        )

    async def delete_api_gateway(self, name: str, project_name: str = None):
        headers = {}

        if project_name:
            headers[NUCLIO_PROJECT_NAME_HEADER] = project_name

        return await self._send_request_to_api(
            method="DELETE",
            path=NUCLIO_API_GATEWAYS_ENDPOINT_TEMPLATE.format(api_gateway=""),
            headers=headers,
            json={"metadata": {"name": name}},
        )

    def _set_iguazio_labels(self, nuclio_object, project_name):
        nuclio_object.metadata.labels[NUCLIO_PROJECT_NAME_LABEL] = project_name
        nuclio_object.metadata.labels[MLRUN_CREATED_LABEL] = "true"

    async def _ensure_async_session(self):
        if not self._session:
            self._session = mlrun.utils.AsyncClientWithRetry(
                retry_on_exception=mlrun.mlconf.httpdb.projects.retry_leader_request_on_exception
                == mlrun.common.schemas.HTTPSessionRetryMode.enabled.value,
                logger=logger,
            )

    async def _close_session(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def _send_request_to_api(
        self, method, path="/", error_message: str = "", **kwargs
    ):
        await self._ensure_async_session()
        response = await self._session.request(
            method=method,
            url=urllib.parse.urljoin(self._nuclio_dashboard_url, path),
            auth=self._auth,
            verify_ssl=False,
            **kwargs,
        )
        if not response:
            return
        if not response.ok:
            try:
                response_body = await response.json()
            except Exception:
                response_body = {}
            self._handle_error_response(
                method,
                self._nuclio_dashboard_url,
                path,
                response,
                response_body,
                error_message,
                kwargs,
            )
        else:
            if response.status == HTTPStatus.NO_CONTENT:
                return
            return await response.json()

    def _handle_error_response(
        self, method, url, path, response, response_body, error_message, kwargs
    ):
        log_kwargs = copy.deepcopy(kwargs)
        log_kwargs.pop("json", None)
        log_kwargs.update({"method": method, "url": url, "path": path})

        try:
            error = response_body.get("error", "")
            error_stack_trace = response_body.get("errorStackTrace", "")
        except Exception:
            pass
        else:
            if error:
                error_message = f"{error_message}: {str(error)}"
            if error:
                log_kwargs.update(
                    {"error": error, "errorStackTrace": error_stack_trace}
                )

        self._logger.warning("Request to nuclio failed. Reason:", **log_kwargs)

        mlrun.errors.raise_for_status(response, error_message)

    def _enrich_nuclio_api_gateway(
        self,
        project_name: str,
        api_gateway_name: str,
        api_gateway: mlrun.common.schemas.APIGateway,
    ) -> mlrun.common.schemas.APIGateway:
        self._set_iguazio_labels(api_gateway, project_name)
        if not api_gateway.spec.host:
            api_gateway.spec.host = (
                f"{api_gateway_name}-{project_name}."
                f"{self._nuclio_domain[self._nuclio_domain.find('.') + 1:]}"
            )

        return api_gateway
