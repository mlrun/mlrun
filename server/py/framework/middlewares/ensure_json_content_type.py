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
from starlette.types import ASGIApp, Receive, Scope, Send

_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class EnsureJSONContentTypeMiddleware:
    app: ASGIApp

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        FastAPI's strict_content_type option (default on since 0.128) only parses a request body as
        JSON when Content-Type: application/json is set. Some clients send pre-serialized JSON
        bodies without that header, which used to parse fine under lenient older FastAPI versions.
        Default a missing Content-Type to application/json for body-bearing requests so those keep
        working — this intentionally reverts strict_content_type's protection for back-compat;
        FastAPI <0.128 had no such protection either, so this is parity, not a new hole. Requests
        with no body, or that already declare a content type (multipart/form uploads included),
        are left untouched.
        """
        if scope["type"] == "http" and scope.get("method") in _BODY_METHODS:
            headers = MutableHeaders(scope=scope)
            if not headers.get("content-type") and _has_body(headers):
                headers["content-type"] = "application/json"

        return await self.app(scope, receive, send)


def _has_body(headers: MutableHeaders) -> bool:
    return (
        headers.get("content-length") not in (None, "", "0")
        or "transfer-encoding" in headers
    )
