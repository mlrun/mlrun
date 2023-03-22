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
import contextlib
import copy
import http.cookies
import typing
import urllib.parse

import aiohttp
import fastapi

import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


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
        self._session: typing.Optional[mlrun.utils.AsyncClientWithRetry] = None
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

    async def trigger_migrations(
        self, request: fastapi.Request = None
    ) -> fastapi.Response:
        """
        only chief can execute migrations
        """
        return await self._proxy_request_to_chief(
            "POST", "operations/migrations", request
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

    async def delete_project(self, name, request: fastapi.Request) -> fastapi.Response:
        """
        delete project can be responsible for deleting schedules. Schedules are running only on chief,
        that is why we re-route requests to chief
        """
        # timeout is greater than default as delete project can take a while because it deletes all the
        # project resources (depends on the deletion strategy)
        return await self._proxy_request_to_chief(
            "DELETE", f"projects/{name}", request, timeout=120
        )

    async def get_clusterization_spec(
        self, return_fastapi_response: bool = True, raise_on_failure: bool = False
    ) -> typing.Union[fastapi.Response, mlrun.api.schemas.ClusterizationSpec]:
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
                return await self._convert_requests_response_to_fastapi_response(
                    chief_response
                )

            return mlrun.api.schemas.ClusterizationSpec(**(await chief_response.json()))

    async def _proxy_request_to_chief(
        self,
        method,
        path,
        request: fastapi.Request = None,
        json: dict = None,
        raise_on_failure: bool = False,
        **kwargs,
    ) -> fastapi.Response:
        request_kwargs = self._resolve_request_kwargs_from_request(
            request, json, **kwargs
        )

        async with self._send_request_to_api(
            method=method,
            path=path,
            raise_on_failure=raise_on_failure,
            **request_kwargs,
        ) as chief_response:
            return await self._convert_requests_response_to_fastapi_response(
                chief_response
            )

    @staticmethod
    def _resolve_request_kwargs_from_request(
        request: fastapi.Request = None, json: dict = None, **kwargs
    ) -> dict:
        request_kwargs = {}
        if request:
            json = json if json else {}
            request_kwargs.update({"json": json})
            request_kwargs.update({"headers": dict(request.headers)})
            request_kwargs.update({"params": dict(request.query_params)})
            request_kwargs.update({"cookies": request.cookies})

        # mask clients host with worker's host
        origin_host = request_kwargs.get("headers", {}).pop("host", None)
        if origin_host:
            # original host requested by client
            request_kwargs["headers"]["x-forwarded-host"] = origin_host

        # let the http client calculate it itself
        # or we will hit serious issues with reverse-proxying (client<->worker<->chief) requests
        request_kwargs.get("headers", {}).pop("content-length", None)

        for cookie_name in list(request_kwargs.get("cookies", {}).keys()):

            # defensive programming - to avoid setting reserved cookie names and explode
            # e.g.: when setting "domain" cookie, it will explode, see python internal http client for more details.
            if http.cookies.Morsel().isReservedKey(cookie_name):
                del request_kwargs["cookies"][cookie_name]

            # iguazio auth cookies might include special characters. to ensure the http client wont escape them
            # we will url-encode them (aka quote), so the value would be safe against such escaping.
            # e.g.: instead of having "x":"y" being escaped to "\"x\":\"y\"", it will be escaped to "%22x%22:%22y%22"
            elif cookie_name == "session" and mlrun.mlconf.is_running_on_iguazio():

                # unquote first, to avoid double quoting ourselves, in case the cookie is already quoted
                unquoted_session = urllib.parse.unquote(
                    request_kwargs["cookies"][cookie_name]
                )
                request_kwargs["cookies"][cookie_name] = urllib.parse.quote(
                    unquoted_session
                )

        request_kwargs.update(**kwargs)
        return request_kwargs

    @staticmethod
    async def _convert_requests_response_to_fastapi_response(
        chief_response: aiohttp.ClientResponse,
    ) -> fastapi.Response:
        # based on the way we implemented the exception handling for endpoints in MLRun we can expect the media type
        # of the response to be of type application/json, see mlrun.api.http_status_error_handler for reference
        return fastapi.responses.Response(
            content=await chief_response.text(),
            status_code=chief_response.status,
            headers=dict(
                chief_response.headers
            ),  # chief_response.headers is of type CaseInsensitiveDict
            media_type="application/json",
        )

    @contextlib.asynccontextmanager
    async def _send_request_to_api(
        self, method, path, raise_on_failure: bool = False, **kwargs
    ) -> aiohttp.ClientResponse:
        await self._ensure_session()
        url = f"{self._api_url}/api/{mlrun.mlconf.api_base_version}/{path}"
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (
                mlrun.mlconf.httpdb.clusterization.worker.request_timeout or 20
            )
        logger.debug("Sending request to chief", method=method, url=url, **kwargs)
        response = None
        try:
            response = await self._session.request(
                method, url, verify_ssl=False, **kwargs
            )
            if not response.ok:
                await self._on_request_api_failure(
                    method, path, response, raise_on_failure, **kwargs
                )
            else:
                logger.debug(
                    "Request to chief succeeded",
                    method=method,
                    url=url,
                    **kwargs,
                )
            yield response
        finally:
            if response:
                response.release()

    async def _ensure_session(self):
        if not self._session:
            self._session = mlrun.utils.AsyncClientWithRetry(
                # This client handles forwarding requests from worker to chief.
                # if we receive 5XX error, the code will be returned to the client.
                #  if client is the SDK - it will handle and retry the request itself, upon its own retry policy
                #  if the client is UI  - it will propagate the error to the user.
                # Thus, do not retry.
                # only exceptions (e.g.: connection initiating).
                raise_for_status=False,
            )

            # if we go any HTTP response, return it, do not retry.
            # by returning `True`, we tell the client the response is "legit" and so, it returns it to its callee.
            self._session.retry_options.evaluate_response_callback = lambda _: True

    async def _on_request_api_failure(
        self, method, path, response, raise_on_failure, **kwargs
    ):
        log_kwargs = copy.deepcopy(kwargs)
        log_kwargs.update({"method": method, "path": path})
        log_kwargs.update(
            {
                "status_code": response.status,
                "reason": response.reason,
                "real_url": response.real_url,
            }
        )
        if response.content:
            try:
                data = await response.json()
                error = data.get("error")
                error_stack_trace = data.get("errorStackTrace")
            except Exception:
                pass
            else:
                log_kwargs.update(
                    {"error": error, "error_stack_trace": error_stack_trace}
                )
        logger.warning("Request to chief failed", **log_kwargs)
        if raise_on_failure:
            mlrun.errors.raise_for_status(response)
