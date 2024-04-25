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

import mlrun.common.schemas


class EnsureBackendVersionMiddleware:
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
        This middleware ensures response header includes backend version
        """
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.append(
                    mlrun.common.schemas.constants.HeaderNames.backend_version,
                    self._backend_version,
                )
            await send(message)

        return await self.app(scope, receive, send_wrapper)
