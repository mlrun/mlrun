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

import re
import time

from starlette.types import Message
from uvicorn._types import (
    ASGIReceiveCallable,
    ASGISendCallable,
    Scope,
)

import framework.utils.telemetry.rest_metrics
from .base import BaseHTTPMiddleware, is_response_start

# Noise endpoints excluded from metrics (mirrors RequestLoggerMiddleware). K8s
# liveness/readiness probes hit /api/healthz constantly and would otherwise
# dominate the request count.
_SILENT_PATH_SUBSTRINGS = ("healthz",)

# Leading MLRun path prefix: optional /api then an optional /vN version segment.
_PATH_PREFIX = re.compile(r"^/api(?:/v\d+)?")


def parse_resource_and_project(path: str) -> tuple[str, str]:
    """Extract the object type and (if any) project a route operates on.

    Returns ``(resource, project)`` as bounded, low-cardinality labels — the
    resource is always a route collection name, never a variable ``{name}``/
    ``{uid}``. ``project`` is "" for non-project-scoped routes.

    Examples::

        /api/v1/projects/{project}/functions/{name} -> ("functions", "{project}")
        /api/v1/projects/{project}                   -> ("projects", "{project}")
        /api/v1/projects                             -> ("projects", "")
        /api/v1/runs                                 -> ("runs", "")
    """
    stripped = _PATH_PREFIX.sub("", path)
    segments = [segment for segment in stripped.split("/") if segment]
    if not segments:
        return "", ""
    if segments[0] == "projects":
        if len(segments) >= 3:
            # /projects/{project}/{resource}/...
            return segments[2], segments[1]
        # /projects or /projects/{project}
        return "projects", (segments[1] if len(segments) == 2 else "")
    return segments[0], ""


class RestMetricsMiddleware(BaseHTTPMiddleware):
    """
    Measures how long each REST call took to process and records it to the
    OpenTelemetry request-duration histogram (see
    ``framework.utils.telemetry.rest_metrics``).
    """

    async def _handle_http(
        self, scope: "Scope", receive: "ASGIReceiveCallable", send: "ASGISendCallable"
    ) -> None:
        start_time = time.perf_counter_ns()
        path = scope["path"]
        should_record = not any(
            substring in path for substring in _SILENT_PATH_SUBSTRINGS
        )

        async def send_wrapper(message: Message) -> None:
            await send(message)
            # http.response.start carries the status and is sent once, before
            # any body — the point at which processing time is complete.
            if should_record and is_response_start(message):
                resource, project = parse_resource_and_project(path)
                framework.utils.telemetry.rest_metrics.record_duration(
                    duration_ms=self._elapsed_time_ms(start_time),
                    method=scope["method"],
                    status_code=message["status"],
                    resource=resource,
                    project=project,
                )

        await self.app(scope, receive, send_wrapper)
