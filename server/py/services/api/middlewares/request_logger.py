# Copyright 2024 Iguazio
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

import time
import traceback
import uuid

import uvicorn.protocols.utils
from starlette.datastructures import MutableHeaders
from starlette.types import Message
from uvicorn._types import (
    ASGI3Application,
    ASGIReceiveCallable,
    ASGISendCallable,
    Scope,
)

import mlrun
import mlrun.common.schemas
from mlrun.utils.logger import Logger


class RequestLoggerMiddleware:
    def __init__(
        self,
        app: "ASGI3Application",
        logger: Logger,
    ) -> None:
        self.app = app
        self._logger = logger
        self._silent_logging_paths = [
            "healthz",
        ]

    async def __call__(
        self, scope: "Scope", receive: "ASGIReceiveCallable", send: "ASGISendCallable"
    ) -> None:
        """
        This middleware logs request and response information for audit and debugging purposes.
        """
        if scope["type"] not in ("http",):
            return await self.app(scope, receive, send)

        headers = MutableHeaders(scope=scope)
        request_id = headers.get("x-request-id") or str(uuid.uuid4())
        # limit request id to 36 characters (uuid4 length) to avoid log lines being too long
        request_id = request_id[:36]
        path_with_query_string = uvicorn.protocols.utils.get_path_with_query_string(
            scope
        )
        scope.setdefault("state", {}).setdefault("request_id", request_id)
        start_time = time.perf_counter_ns()
        should_log = not any(
            silent_logging_path in path_with_query_string
            for silent_logging_path in self._silent_logging_paths
        )
        if should_log:
            self._logger.debug(
                "Received request",
                headers=self._log_headers(headers),
                method=scope["method"],
                client_address=self._resolve_client_address(scope),
                http_version=scope["http_version"],
                request_id=request_id,
                uri=path_with_query_string,
            )

        async def send_wrapper(message: Message) -> None:
            try:
                await send(message)
            finally:
                if message["type"] == "http.response.start":
                    # convert from nanoseconds to milliseconds
                    elapsed_time_in_ms = (
                        (time.perf_counter_ns() - start_time) / 1000 / 1000
                    )
                    if should_log:
                        self._logger.debug(
                            "Sending response",
                            status_code=message["status"],
                            request_id=request_id,
                            elapsed_time_in_ms=elapsed_time_in_ms,
                            uri=path_with_query_string,
                            method=scope["method"],
                            headers=self._log_headers(headers),
                        )

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            self._logger.warning(
                "Request handling failed. Sending response",
                # User middleware (like this one) runs after the exception handling middleware,
                # the only thing running after it is starletter's ServerErrorMiddleware which is responsible
                # for catching any un-handled exception and transforming it to 500 response.
                # therefore we can statically assign status code to 500
                status_code=500,
                request_id=request_id,
                uri=path_with_query_string,
                method=scope["method"],
                exc=mlrun.errors.err_to_str(exc),
                traceback=traceback.format_exc(),
            )
            raise

    def _resolve_client_address(self, scope):
        # uvicorn expects this to be a tuple while starlette test client sets it to be a list
        if isinstance(scope.get("client"), list):
            scope["client"] = tuple(scope.get("client"))
        return uvicorn.protocols.utils.get_client_addr(scope)

    def _log_headers(self, headers: MutableHeaders):
        headers_to_log = headers.mutablecopy()
        headers_to_omit = [
            "authorization",
            "cookie",
            "x-v3io-session-key",
            "x-v3io-access-key",
        ]
        for name, values in headers.items():
            if name in headers_to_omit:
                del headers_to_log[name]
        return dict(headers_to_log.items())
