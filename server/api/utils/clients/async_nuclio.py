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
from typing import Union

import aiohttp

import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes.api_gateway
import mlrun.utils
from mlrun.utils import logger

NUCLIO_API_SESSIONS_ENDPOINT = "/api/sessions/"
NUCLIO_API_GATEWAYS_ENDPOINT = "/api/api_gateways/"
API_GATEWAY_NAMESPACE_HEADER = "X-Nuclio-Api-Gateway-Namespace"
NUCLIO_PROJECT_NAME_HEADER = "X-Nuclio-Project-Name"


class Client:
    def __init__(self, auth_info: mlrun.common.schemas.AuthInfo):
        self._session = None
        self._auth = aiohttp.BasicAuth(auth_info.username, auth_info.session)
        self._logger = logger.get_child("nuclio-client")
        self._nuclio_dashboard_url = mlrun.mlconf.nuclio_dashboard_url
        self._nuclio_domain = urllib.parse.urlparse(self._nuclio_dashboard_url).netloc

    async def list_api_gateways(self, project_name=None):
        headers = {}

        if project_name:
            headers[NUCLIO_PROJECT_NAME_HEADER] = project_name

        return await self._send_request_to_api(
            method="GET",
            url=self._nuclio_dashboard_url,
            path=NUCLIO_API_GATEWAYS_ENDPOINT.format(api_gateway=""),
            headers=headers,
        )

    async def create_api_gateway(
        self,
        project_name: str,
        api_gateway_name: str,
        functions: list,
        host: Union[str, None] = None,
        path="/",
        description="",
        username: Union[str, None] = None,
        password: Union[str, None] = None,
        canary: Union[list, None] = None,
    ):
        headers = {}

        if project_name:
            headers[NUCLIO_PROJECT_NAME_HEADER] = project_name

        body = self._generate_nuclio_api_gateway_body(
            project_name=project_name,
            api_gateway_name=api_gateway_name,
            functions=functions,
            host=host,
            path=path,
            description=description,
            username=username,
            password=password,
            canary=canary,
        )

        return await self._send_request_to_api(
            method="POST",
            url=self._nuclio_dashboard_url,
            path=NUCLIO_API_GATEWAYS_ENDPOINT,
            headers=headers,
            json=body,
        )

    async def _ensure_async_session(self):
        if not self._session:
            self._session = mlrun.utils.AsyncClientWithRetry(
                retry_on_exception=mlrun.mlconf.httpdb.projects.retry_leader_request_on_exception
                == mlrun.common.schemas.HTTPSessionRetryMode.enabled.value,
                logger=logger,
            )

    async def _send_request_to_api(
        self, method, url, path="/", error_message: str = "", **kwargs
    ):
        await self._ensure_async_session()
        response = await self._session.request(
            method=method,
            url=urllib.parse.urljoin(url, path),
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
                method, url, path, response, response_body, error_message, kwargs
            )
        else:
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

    def _generate_nuclio_api_gateway_body(
        self,
        project_name,
        api_gateway_name,
        functions,
        host,
        path,
        description="",
        username=None,
        password=None,
        canary=None,
    ) -> dict:
        if not functions:
            raise ValueError("functions should contain at least one object")
        host = (
            f"{api_gateway_name}-{project_name}.{self._nuclio_domain[self._nuclio_domain.find('.')+1:]}"
            if not host
            else host
        )

        authentication_mode = (
            mlrun.runtimes.api_gateway.NO_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE
            if not username and not password
            else mlrun.runtimes.api_gateway.BASIC_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE
        )
        body = {
            "spec": {
                "name": api_gateway_name,
                "description": description,
                "path": path,
                "authenticationMode": authentication_mode,
                "upstreams": [
                    {
                        "kind": "nucliofunction",
                        "nucliofunction": {
                            "name": functions[0],
                        },
                        "percentage": 0,
                    }
                ],
                "host": host,
            },
            "metadata": {
                "labels": {
                    "nuclio.io/project-name": project_name,
                },
                "name": api_gateway_name,
            },
        }

        # Handle authentication info
        if (
            authentication_mode
            == mlrun.runtimes.api_gateway.BASIC_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE
        ):
            if username and password:
                body["spec"]["authentication"] = {
                    "basicAuth": {
                        "username": username,
                        "password": password,
                    }
                }
            else:
                raise mlrun.errors.MLRunPreconditionFailedError(
                    "basicAuth authentication requires username and " "password"
                )

        # Handle canary function info
        if canary:
            if len(canary) != len(functions):
                raise ValueError(
                    "Functions list should be the same length as canary list"
                )
            upstream = [
                {
                    "kind": "nucliofunction",
                    "nucliofunction": {"name": function_name},
                    "percentage": percentage,
                }
                for function_name, percentage in zip(functions, canary)
            ]
            body["spec"]["upstreams"] = upstream

        return body
