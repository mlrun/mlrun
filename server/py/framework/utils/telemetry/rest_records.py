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

"""Per-REST-call OTel log records for MLRun API-bearing services.

Emits one structured log record per request, carrying all call fields needed
for ad-hoc debugging and audit. Records supplement the always-on histogram
metrics with full-detail rows.

Gated behind telemetry.enabled + telemetry.rest_metrics.enabled; init() is a
no-op and emit_record() short-circuits when telemetry is off.

Sampling (telemetry.rest_metrics.sample_rate): routine calls recorded at that
probability; failed/slow/large calls are always kept.
"""

import http
import random

from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,  # noqa: F401
)

# The OTel Python logs SDK is still under a private path in SDK 1.42.
from opentelemetry.sdk._logs import LoggerProvider

import mlrun
import mlrun.errors
import mlrun.utils

import framework.utils.telemetry.otel

_NON_SUCCESS_STATUS = http.HTTPStatus.MULTIPLE_CHOICES  # first non-2xx code
_SLOW_THRESHOLD_SECONDS = 10
_LARGE_RESPONSE_KIB = 100

_provider: LoggerProvider | None = None
_otel_logger = None  # opentelemetry.sdk._logs._internal.Logger


def is_enabled() -> bool:
    """True once init() succeeds; False before init or after shutdown()."""
    return _provider is not None


def init(service_name: str) -> None:
    """Wire up the REST-records LoggerProvider.

    No-op when already initialized. Does not set the global logger provider.

    :param service_name: Bare service name (e.g. ``"api"``); ``"mlrun-"`` is prepended.
    """
    global _provider, _otel_logger
    if _provider is not None:
        mlrun.utils.logger.warning(
            "REST records telemetry already initialized; skipping re-init"
        )
        return
    _provider = framework.utils.telemetry.otel.build_log_provider(
        service_name=f"mlrun-{service_name}"
    )
    _otel_logger = _provider.get_logger("mlrun.rest_records")
    mlrun.utils.logger.info(
        "REST records telemetry initialized",
        service_name=service_name,
        otlp_endpoint=mlrun.mlconf.telemetry.otlp_endpoint,
    )


def shutdown() -> None:
    """Flush pending records and tear down the LoggerProvider.

    No-op when telemetry was never initialized.
    """
    global _provider, _otel_logger
    if _provider is None:
        return
    try:
        _provider.shutdown()
        mlrun.utils.logger.info("REST records telemetry flushed and torn down")
    except Exception as exc:
        mlrun.utils.logger.warning(
            "REST records telemetry shutdown failed",
            error=mlrun.errors.err_to_str(exc),
        )
    finally:
        _provider = None
        _otel_logger = None


def should_sample_record(
    *,
    status_code: int,
    elapsed_seconds: float,
    response_size_kib: float,
) -> bool:
    """Return True if this call's log record should be emitted.

    Failed, slow, or large calls are always kept. Everything else is sampled
    at ``telemetry.rest_metrics.sample_rate`` probability.

    :param status_code:       HTTP response status code.
    :param elapsed_seconds:   Processing time in seconds.
    :param response_size_kib: Response body size in kibibytes.
    :return: Whether the call's log record should be emitted.
    """
    if (
        status_code >= _NON_SUCCESS_STATUS
        or elapsed_seconds > _SLOW_THRESHOLD_SECONDS
        or response_size_kib > _LARGE_RESPONSE_KIB
    ):
        return True
    return random.random() < mlrun.mlconf.telemetry.rest_metrics.sample_rate


def emit_record(
    *,
    path: str,
    query_string: str,
    method: str,
    status_code: int,
    duration_ms: float,
    request_size_bytes: int,
    response_size_bytes: int,
    resource: str,
    project: str,
    client_ip: str,
    request_id: str,
    item_count: int | None,
) -> None:
    """Emit an OTel log record for a completed REST call. No-op when not initialized.

    :param path:               Request path (without query string).
    :param query_string:       URL query string.
    :param method:             HTTP method or ``"LIST"`` for collection GETs.
    :param status_code:        HTTP response status code.
    :param duration_ms:        Processing time in milliseconds.
    :param request_size_bytes: Request body size in bytes.
    :param response_size_bytes: Response body size in bytes.
    :param resource:           Object type (e.g. ``"functions"``).
    :param project:            Project name, or ``""`` for non-scoped routes.
    :param client_ip:          Client IP address.
    :param request_id:         Request ID from headers, or ``""``.
    :param item_count:         Objects returned (list calls only), or ``None``.
    """
    if _otel_logger is None:
        return
    url = f"{path}?{query_string}" if query_string else path
    attributes: dict = {
        "system_id": mlrun.mlconf.system_id or "",
        "method": method,
        "status_code": status_code,
        "resource": resource,
        "project": project,
        "client_ip": client_ip,
        "request_id": request_id,
        "url": url,
        "request_size_bytes": request_size_bytes,
        "response_size_bytes": response_size_bytes,
        "elapsed_ms": duration_ms,
    }
    if item_count is not None:
        attributes["items_returned"] = item_count
    severity = (
        SeverityNumber.ERROR
        if status_code >= 500
        else SeverityNumber.WARN
        if status_code >= 400
        else SeverityNumber.INFO
    )
    _otel_logger.emit(
        body=f"{method} {url} {status_code}",
        severity_number=severity,
        attributes=attributes,
    )
