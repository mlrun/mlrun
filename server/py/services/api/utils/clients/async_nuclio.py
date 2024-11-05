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
import urllib.parse
from http import HTTPStatus

import aiohttp

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.utils
from mlrun.common.helpers import generate_api_gateway_name
from mlrun.utils import logger

NUCLIO_API_SESSIONS_ENDPOINT = "/api/sessions/"
NUCLIO_API_GATEWAYS_ENDPOINT_TEMPLATE = "/api/api_gateways/{api_gateway}"
NUCLIO_API_GATEWAY_NAMESPACE_HEADER = "X-Nuclio-Api-Gateway-Namespace"
NUCLIO_DELETE_FUNCTIONS_WITH_API_GATEWAYS_HEADER = (
    "X-Nuclio-Delete-Function-With-API-Gateways"
)
NUCLIO_FUNCTIONS_ENDPOINT_TEMPLATE = "/api/functions/{function}"
NUCLIO_PROJECT_NAME_HEADER = "X-Nuclio-Project-Name"


class Client:
    def __init__(self, auth_info: mlrun.common.schemas.AuthInfo):
        self._session = None
        login = auth_info.username
        self._auth = aiohttp.BasicAuth(login, auth_info.session) if login else None
        self._logger = logger.get_child("nuclio-client")
        self._nuclio_dashboard_url = mlrun.mlconf.nuclio_dashboard_url

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
            parsed_api_gateways[name] = mlrun.common.schemas.APIGateway.parse_obj(
                gw
            ).replace_nuclio_names_with_mlrun_names()
        return parsed_api_gateways

    async def api_gateway_exists(self, name: str, project_name: str = None):
        # enrich api gateway name with project prefix
        name = generate_api_gateway_name(project_name, name)

        return name in await self.list_api_gateways(project_name=project_name)

    async def get_api_gateway(self, name: str, project_name: str = None):
        headers = {}

        # enrich api gateway name with project prefix
        name = generate_api_gateway_name(project_name, name)

        if project_name:
            headers[NUCLIO_PROJECT_NAME_HEADER] = project_name

        api_gateway = await self._send_request_to_api(
            method="GET",
            path=NUCLIO_API_GATEWAYS_ENDPOINT_TEMPLATE.format(api_gateway=name),
            headers=headers,
        )
        return mlrun.common.schemas.APIGateway.parse_obj(
            api_gateway
        ).replace_nuclio_names_with_mlrun_names()

    async def store_api_gateway(
        self,
        project_name: str,
        api_gateway: mlrun.common.schemas.APIGateway,
        create: bool = False,
    ):
        headers = {}
        self._enrich_nuclio_api_gateway(
            project_name=project_name,
            api_gateway=api_gateway,
        )

        if project_name:
            headers[NUCLIO_PROJECT_NAME_HEADER] = project_name

        body = api_gateway.dict(exclude_none=True)
        method = "POST" if create else "PUT"
        path = (
            NUCLIO_API_GATEWAYS_ENDPOINT_TEMPLATE.format(
                api_gateway=api_gateway.spec.name
            )
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

        # enrich api gateway name with project prefix
        name = generate_api_gateway_name(project_name, name)

        if project_name:
            headers[NUCLIO_PROJECT_NAME_HEADER] = project_name

        return await self._send_request_to_api(
            method="DELETE",
            path=NUCLIO_API_GATEWAYS_ENDPOINT_TEMPLATE.format(api_gateway=""),
            headers=headers,
            json={"metadata": {"name": name}},
        )

    async def delete_function(self, name: str, project_name: str = None):
        # this header allows nuclio to delete function along with its api gateways
        headers = {NUCLIO_DELETE_FUNCTIONS_WITH_API_GATEWAYS_HEADER: "true"}

        if project_name:
            headers[NUCLIO_PROJECT_NAME_HEADER] = project_name

        return await self._send_request_to_api(
            method="DELETE",
            path=NUCLIO_FUNCTIONS_ENDPOINT_TEMPLATE.format(function=""),
            headers=headers,
            json={"metadata": {"name": name}},
        )

    def _set_iguazio_labels(self, nuclio_object, project_name):
        nuclio_object.metadata.labels[
            mlrun_constants.MLRunInternalLabels.nuclio_project_name
        ] = project_name
        nuclio_object.metadata.labels[mlrun_constants.MLRunInternalLabels.created] = (
            "true"
        )

    async def _ensure_async_session(self):
        if not self._session:
            self._session = mlrun.utils.AsyncClientWithRetry(
                raise_for_status=False,
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
                error_message = (
                    f"{error_message}: {str(error)}" if error_message else str(error)
                )
            if error:
                log_kwargs.update(
                    {"error": error, "errorStackTrace": error_stack_trace}
                )

        self._logger.warning("Request to nuclio failed. Reason:", **log_kwargs)

        mlrun.errors.raise_for_status(response, error_message)

    def _enrich_nuclio_api_gateway(
        self,
        project_name: str,
        api_gateway: mlrun.common.schemas.APIGateway,
    ) -> mlrun.common.schemas.APIGateway:
        self._set_iguazio_labels(api_gateway, project_name)
        api_gateway.enrich_mlrun_names()
        return api_gateway
