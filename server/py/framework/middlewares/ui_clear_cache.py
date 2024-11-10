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
from mlrun.config import config


class UiClearCacheMiddleware:
    def __init__(
        self,
        app: "ASGI3Application",
        backend_version: str,
    ) -> None:
        self.app = app
        self._backend_version = backend_version

    async def __call__(
        self, scope: "Scope", receive: "ASGIReceiveCallable", send: "ASGISendCallable"
    ) -> None:
        """
        This middleware tells ui when to clear its cache based on backend version changes.
        """
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        # ask ui to reload cache if
        #  - ui sent a version
        #  - backend is not a development version
        #  - ui version is different from backend version
        # otherwise, do not ask ui to reload its cache as it will make each request to reload ui and clear cache
        request_headers = MutableHeaders(scope=scope)
        ui_version = request_headers.get(
            mlrun.common.schemas.constants.HeaderNames.ui_version, ""
        )

        async def send_wrapper(message: Message) -> None:
            if (
                message["type"] == "http.response.start"
                and ui_version
                and ui_version != config.version
                and not self._is_development_version()
            ):
                response_headers = MutableHeaders(scope=message)

                # clear site cache
                response_headers.append("Clear-Site-Data", '"cache"')
                # tell ui to reload
                response_headers.append(
                    mlrun.common.schemas.constants.HeaderNames.ui_clear_cache,
                    "true",
                )
            await send(message)

        return await self.app(scope, receive, send_wrapper)

    def _is_development_version(self):
        return self._backend_version.startswith("0.0.0")
