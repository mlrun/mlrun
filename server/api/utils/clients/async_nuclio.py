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

import aiohttp

import mlrun.common.schemas
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

    async def list_api_gateways(self, project_name=None):
        headers = {}

        if project_name:
            headers[NUCLIO_PROJECT_NAME_HEADER] = project_name

        return await self._send_request_to_api(
            method="GET",
            url=self._nuclio_dashboard_url,
            path=NUCLIO_API_GATEWAYS_ENDPOINT,
            headers=headers,
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
