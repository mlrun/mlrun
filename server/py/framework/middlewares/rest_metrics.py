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

import http
import json
import re
import time

from starlette.types import Message
from uvicorn._types import (
    ASGIReceiveCallable,
    ASGISendCallable,
    Scope,
)

import mlrun.errors
import mlrun.utils

import framework.utils.telemetry.rest_metrics
from .base import BaseHTTPMiddleware, is_response_start

# Noise endpoints excluded from metrics (mirrors RequestLoggerMiddleware). K8s
# liveness/readiness probes hit /api/healthz constantly and would otherwise
# dominate the request count.
_SILENT_PATH_SUBSTRINGS = ("healthz",)

# Leading MLRun path prefix: optional /api then an optional /vN version segment.
_PATH_PREFIX = re.compile(r"^/api(?:/v\d+)?")

# Binary (2^10), matching the size histograms' "KiBy" (kibibyte) OTel unit —
# see telemetry/rest_metrics.py's init() for why the name and unit must agree.
_BYTES_PER_KIBIBYTE = 1024


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


def parse_get_vs_list(method: str, scope: "Scope") -> str:
    """Distinguish a single-object GET from a collection-returning GET.

    Defaults every GET to "get" and only promotes to "list" when the matched
    endpoint function's name starts with ``list_`` — the one signal this
    codebase applies consistently to every genuine collection endpoint
    (``list_runs``, ``list_artifact_tags``, ``list_pipelines``, ...). Path
    shape alone doesn't work: plenty of singleton/action endpoints have a
    literal trailing segment without returning a collection (``/build/status``,
    ``/client-spec``, ``.../drift-over-time``, ``.../nuclio/{name}/deploy``),
    and their function names don't follow any ``get_``/``list_`` convention
    either (``build_status``, ``clusterization_spec``) — so requiring an
    explicit ``get_`` match would leave them unclassified. The matched route
    (and its endpoint function) is stamped onto ``scope["route"]`` by
    FastAPI's router before the endpoint runs, so it's available once
    ``http.response.start`` fires. Only GET calls are classified; everything
    else (and unmatched/404 routes, which never get a ``route``) returns "".

    :param method: HTTP method (e.g. ``GET``).
    :param scope:  The ASGI request scope, post-routing.
    :return: A ``GetVsList`` member, or "" when not applicable.
    """
    if method != http.HTTPMethod.GET:
        return ""
    route = scope.get("route")
    endpoint_name = getattr(getattr(route, "endpoint", None), "__name__", "")
    if not endpoint_name:
        return ""
    get_vs_list = framework.utils.telemetry.rest_metrics.GetVsList
    return get_vs_list.LIST if endpoint_name.startswith("list_") else get_vs_list.GET


def parse_item_count(body: bytes) -> int | None:
    """Count the objects returned by a list call from its JSON response body.

    List endpoints have no shared response envelope — the collection lives
    under a different key per resource (``runs``, ``artifacts``, ``funcs``,
    ...). Rather than hard-coding every key, this takes the first top-level
    list-valued field, which holds for every list endpoint in this codebase.

    :param body: The full, concatenated response body.
    :return: The number of items found, or None if the body isn't a JSON
             object/array or has no list-valued field to count.
    """
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                return len(value)
    return None


class RestMetricsMiddleware(BaseHTTPMiddleware):
    """
    Measures how long each REST call took to process, its request/response
    body sizes, and (for list calls) how many objects it returned, then
    records them to the OpenTelemetry instruments in
    ``framework.utils.telemetry.rest_metrics``.

    Everything is recorded together at the final ``http.response.body``
    message, once every value is available — request/response size, item
    count, and duration (covering the full time to deliver the response, not
    just time-to-first-byte). ``should_sample`` is evaluated exactly once per
    call, from that single, complete picture, so a call is either kept for
    every instrument or dropped for every instrument — never a mix.
    """

    async def _handle_http(
        self, scope: "Scope", receive: "ASGIReceiveCallable", send: "ASGISendCallable"
    ) -> None:
        start_time = time.perf_counter_ns()
        path = scope["path"]
        should_record = not any(
            substring in path for substring in _SILENT_PATH_SUBSTRINGS
        )

        request_size_bytes = 0

        async def receive_wrapper() -> "Message":
            nonlocal request_size_bytes
            message = await receive()
            if should_record and message["type"] == "http.request":
                request_size_bytes += len(message.get("body") or b"")
            return message

        # Mutated by send_wrapper across calls; only meaningful once
        # http.response.start has been observed.
        response_state = {
            "status_code": None,
            "get_vs_list": "",
            "response_size_bytes": 0,
            "response_body": bytearray(),
        }

        async def send_wrapper(message: "Message") -> None:
            await send(message)
            if not should_record:
                return
            try:
                if is_response_start(message):
                    response_state["status_code"] = message["status"]
                    response_state["get_vs_list"] = parse_get_vs_list(
                        scope["method"], scope
                    )
                    return
                if message.get("type") != "http.response.body":
                    return
                body = message.get("body") or b""
                response_state["response_size_bytes"] += len(body)
                if (
                    response_state["get_vs_list"]
                    == framework.utils.telemetry.rest_metrics.GetVsList.LIST
                ):
                    response_state["response_body"].extend(body)
                if message.get("more_body", False):
                    # Streamed body still in flight — nothing to record yet.
                    return
                self._record_call(
                    path=path,
                    method=scope["method"],
                    duration_ms=self._elapsed_time_ms(start_time),
                    request_size_bytes=request_size_bytes,
                    response_state=response_state,
                )
            except Exception as exc:
                mlrun.utils.logger.warning(
                    "REST metrics recording failed",
                    path=path,
                    error=mlrun.errors.err_to_str(exc),
                )

        await self.app(scope, receive_wrapper, send_wrapper)

    @staticmethod
    def _record_call(
        *,
        path: str,
        method: str,
        duration_ms: float,
        request_size_bytes: int,
        response_state: dict,
    ) -> None:
        """Decide whether to sample, then record every instrument for one call."""
        resource, project = parse_resource_and_project(path)
        status_code = response_state["status_code"]
        get_vs_list = response_state["get_vs_list"]
        request_size_kib = request_size_bytes / _BYTES_PER_KIBIBYTE
        response_size_kib = response_state["response_size_bytes"] / _BYTES_PER_KIBIBYTE

        if not framework.utils.telemetry.rest_metrics.should_sample(
            status_code=status_code,
            elapsed_seconds=duration_ms / 1000,
            response_size_kib=response_size_kib,
        ):
            return

        framework.utils.telemetry.rest_metrics.record_duration(
            duration_ms=duration_ms,
            method=method,
            status_code=status_code,
            resource=resource,
            project=project,
            get_vs_list=get_vs_list,
        )
        framework.utils.telemetry.rest_metrics.record_request_size(
            size_kib=request_size_kib,
            method=method,
            status_code=status_code,
            resource=resource,
            project=project,
            get_vs_list=get_vs_list,
        )
        framework.utils.telemetry.rest_metrics.record_response_size(
            size_kib=response_size_kib,
            method=method,
            status_code=status_code,
            resource=resource,
            project=project,
            get_vs_list=get_vs_list,
        )
        if get_vs_list == framework.utils.telemetry.rest_metrics.GetVsList.LIST:
            item_count = parse_item_count(bytes(response_state["response_body"]))
            if item_count is not None:
                framework.utils.telemetry.rest_metrics.record_items_returned(
                    item_count=item_count,
                    status_code=status_code,
                    resource=resource,
                    project=project,
                )
