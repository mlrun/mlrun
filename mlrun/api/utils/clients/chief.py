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
import copy
import typing

import fastapi
import requests.adapters

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
        self._session = mlrun.utils.HTTPSessionWithRetry(
            # when the request is forwarded to the chief, if we receive a 5XX error, the code will be forwarded to the
            # client. if the client is the SDK, it will retry the request. if the client is UI, it will receive the
            # error without retry. so no need to retry the request to the chief on status codes, only exceptions for
            # failed handshakes.
            retry_on_status=False,
            verbose=True,
        )
        self._api_url = mlrun.mlconf.resolve_chief_api_url()
        # remove backslash from end of api url
        self._api_url = (
            self._api_url[:-1] if self._api_url.endswith("/") else self._api_url
        )

    def get_internal_background_task(
        self, name: str, request: fastapi.Request = None
    ) -> fastapi.Response:
        """
        internal background tasks are managed by the chief only
        """
        return self._proxy_request_to_chief("GET", f"background-tasks/{name}", request)

    def trigger_migrations(self, request: fastapi.Request = None) -> fastapi.Response:
        """
        only chief can execute migrations
        """
        return self._proxy_request_to_chief("POST", "operations/migrations", request)

    def create_schedule(
        self, project: str, request: fastapi.Request, json: dict
    ) -> fastapi.Response:
        """
        Schedules are running only on chief
        """
        return self._proxy_request_to_chief(
            "POST", f"projects/{project}/schedules", request, json
        )

    def update_schedule(
        self, project: str, name: str, request: fastapi.Request, json: dict
    ) -> fastapi.Response:
        """
        Schedules are running only on chief
        """
        return self._proxy_request_to_chief(
            "PUT", f"projects/{project}/schedules/{name}", request, json
        )

    def delete_schedule(
        self, project: str, name: str, request: fastapi.Request
    ) -> fastapi.Response:
        """
        Schedules are running only on chief
        """
        return self._proxy_request_to_chief(
            "DELETE", f"projects/{project}/schedules/{name}", request
        )

    def delete_schedules(
        self, project: str, request: fastapi.Request
    ) -> fastapi.Response:
        """
        Schedules are running only on chief
        """
        return self._proxy_request_to_chief(
            "DELETE", f"projects/{project}/schedules", request
        )

    def invoke_schedule(
        self, project: str, name: str, request: fastapi.Request
    ) -> fastapi.Response:
        """
        Schedules are running only on chief
        """
        return self._proxy_request_to_chief(
            "POST", f"projects/{project}/schedules/{name}/invoke", request
        )

    def submit_job(self, request: fastapi.Request, json: dict) -> fastapi.Response:
        """
        submit job can be responsible for creating schedules and schedules are running only on chief,
        so when the job contains a schedule, we re-route the request to chief
        """
        return self._proxy_request_to_chief(
            "POST",
            "submit_job",
            request,
            json,
            timeout=int(mlrun.mlconf.submit_timeout),
        )

    def build_function(self, request: fastapi.Request, json: dict) -> fastapi.Response:
        """
        if serving function and track_models is enabled, it means that schedules will be created as part of
        building the function, then we re-route the request to chief
        """
        return self._proxy_request_to_chief("POST", "build/function", request, json)

    def delete_project(self, name, request: fastapi.Request) -> fastapi.Response:
        """
        delete project can be responsible for deleting schedules. Schedules are running only on chief,
        that is why we re-route requests to chief
        """
        # timeout is greater than default as delete project can take a while because it deletes all the
        # project resources (depends on the deletion strategy)
        return self._proxy_request_to_chief(
            "DELETE", f"projects/{name}", request, timeout=120
        )

    def get_clusterization_spec(
        self, return_fastapi_response: bool = True, raise_on_failure: bool = False
    ) -> typing.Union[fastapi.Response, mlrun.api.schemas.ClusterizationSpec]:
        """
        This method is used both for proxying requests from worker to chief and for aligning the worker state
        with the clusterization spec brought from the chief
        """
        chief_response = self._send_request_to_api(
            method="GET",
            path="clusterization-spec",
            raise_on_failure=raise_on_failure,
        )

        if return_fastapi_response:
            return self._convert_requests_response_to_fastapi_response(chief_response)

        return mlrun.api.schemas.ClusterizationSpec(**chief_response.json())

    def _proxy_request_to_chief(
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

        chief_response = self._send_request_to_api(
            method=method,
            path=path,
            raise_on_failure=raise_on_failure,
            **request_kwargs,
        )

        return self._convert_requests_response_to_fastapi_response(chief_response)

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

        request_kwargs.update(**kwargs)
        return request_kwargs

    @staticmethod
    def _convert_requests_response_to_fastapi_response(
        chief_response: requests.Response,
    ) -> fastapi.Response:
        # based on the way we implemented the exception handling for endpoints in MLRun we can expect the media type
        # of the response to be of type application/json, see mlrun.api.http_status_error_handler for reference
        return fastapi.responses.Response(
            content=chief_response.content,
            status_code=chief_response.status_code,
            headers=dict(
                chief_response.headers
            ),  # chief_response.headers is of type CaseInsensitiveDict
            media_type="application/json",
        )

    # TODO change this to use async calls
    def _send_request_to_api(
        self, method, path, raise_on_failure: bool = False, **kwargs
    ):
        url = f"{self._api_url}/api/{mlrun.mlconf.api_base_version}/{path}"
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (
                mlrun.mlconf.httpdb.clusterization.worker.request_timeout or 20
            )
        logger.debug("Sending request to chief", method=method, url=url, **kwargs)
        response = self._session.request(method, url, verify=False, **kwargs)
        if not response.ok:
            log_kwargs = copy.deepcopy(kwargs)
            log_kwargs.update({"method": method, "path": path})
            if response.content:
                try:
                    data = response.json()
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
            return response
        # there are some responses like NO-CONTENT which doesn't return a json body
        try:
            data = response.json()
        except Exception:
            data = response.text
        logger.debug(
            "Request to chief succeeded",
            method=method,
            url=url,
            **kwargs,
            response=data,
        )
        return response
