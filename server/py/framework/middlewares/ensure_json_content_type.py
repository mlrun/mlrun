# Copyright 2026 Iguazio
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

from starlette.datastructures import MutableHeaders
from uvicorn._types import (
    ASGI3Application,
    ASGIReceiveCallable,
    ASGISendCallable,
    Scope,
)


class EnsureJsonContentTypeMiddleware:
    def __init__(
        self,
        app: "ASGI3Application",
    ) -> None:
        self.app = app

    async def __call__(
        self, scope: "Scope", receive: "ASGIReceiveCallable", send: "ASGISendCallable"
    ) -> None:
        """
        FastAPI's strict_content_type option (default on since 0.128) only parses a request body as
        JSON when Content-Type: application/json is set. Some clients send pre-serialized JSON bodies
        without that header, which used to parse fine under lenient older FastAPI versions. Default a
        missing Content-Type to application/json so those requests keep working — this intentionally
        reverts strict_content_type's protection for back-compat; FastAPI <0.128 had no such
        protection either, so this is parity, not a new hole. Requests that already declare a content
        type (multipart/form uploads included, since those always carry their own boundary-bearing
        Content-Type) are left untouched.
        """
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        headers = MutableHeaders(scope=scope)
        headers.setdefault("content-type", "application/json")

        return await self.app(scope, receive, send)
