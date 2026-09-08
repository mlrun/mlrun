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

import abc
import time

# ASGI type hints are sourced from uvicorn's (private) vendored copy rather than
# `asgiref.typing`: asgiref is not a dependency of this repo, and the rest of the
# middlewares package already annotates against `uvicorn._types`.
from starlette.types import Message
from uvicorn._types import (
    ASGI3Application,
    ASGIReceiveCallable,
    ASGISendCallable,
    Scope,
)


def is_response_start(message: "Message") -> bool:
    """
    Whether an outgoing ASGI message is the response-start event, i.e. the one
    carrying the status code and headers, sent once before any response body.

    See the ASGI HTTP spec, "Response Start":
    https://asgi.readthedocs.io/en/latest/specs/www.html
    """
    return message["type"] == "http.response.start"


class BaseHTTPMiddleware(abc.ABC):
    """
    Base class for ASGI middlewares that act only on HTTP requests and need to
    measure per-request processing time.

    It owns the non-HTTP passthrough and the elapsed-time computation so that
    subclasses only implement ``_handle_http`` with what they want to do per call.
    """

    def __init__(self, app: "ASGI3Application") -> None:
        self.app = app

    async def __call__(
        self, scope: "Scope", receive: "ASGIReceiveCallable", send: "ASGISendCallable"
    ) -> None:
        """
        Entry point implementing the ASGI 3.0 single-callable application/middleware
        contract: an async callable of ``(scope, receive, send)``.

        See the ASGI spec, "Applications":
        https://asgi.readthedocs.io/en/latest/specs/main.html#applications
        """
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        return await self._handle_http(scope, receive, send)

    @abc.abstractmethod
    async def _handle_http(
        self, scope: "Scope", receive: "ASGIReceiveCallable", send: "ASGISendCallable"
    ) -> None:
        raise NotImplementedError()

    @staticmethod
    def _elapsed_time_ms(start_time_ns: int) -> float:
        # convert from nanoseconds to milliseconds
        return (time.perf_counter_ns() - start_time_ns) / 1000 / 1000
