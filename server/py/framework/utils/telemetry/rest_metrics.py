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

"""Per-REST-call OTel telemetry for MLRun API-bearing services.

Records duration, request/response size, and items-returned histograms per
REST call, tagged with system_id/method/status_code/resource/project.
Collection-returning GETs report method="LIST" (see parse_method).

Gated behind telemetry.enabled + telemetry.rest_metrics.enabled; init() is a
no-op and record_*() short-circuits when either is off.

Histograms are always emitted for every call — no sampling.
"""

import collections.abc

from opentelemetry.metrics import (
    Histogram,
    Meter,
)
from opentelemetry.sdk.metrics import MeterProvider

import mlrun
import mlrun.errors
import mlrun.utils

import framework.utils.telemetry.otel

# ms, not seconds: the SDK's default buckets (5–10000) map onto real latencies
# in ms. Seconds would pack nearly all requests into the first bucket.
_DURATION_INSTRUMENT_NAME = "mlrun_rest_request_duration_milliseconds"
# "_kibibytes" matches unit="KiBy"'s Prometheus expansion — the exporter only
# skips appending the unit suffix when the name already ends with that word.
# "_kilobytes" would not match "kibibytes", causing a double-suffix.
_REQUEST_SIZE_INSTRUMENT_NAME = "mlrun_rest_request_size_kibibytes"
_RESPONSE_SIZE_INSTRUMENT_NAME = "mlrun_rest_response_size_kibibytes"
_ITEMS_RETURNED_INSTRUMENT_NAME = "mlrun_rest_response_num_items"

_provider: MeterProvider | None = None
_meter: Meter | None = None
_duration_histogram: Histogram | None = None
_request_size_histogram: Histogram | None = None
_response_size_histogram: Histogram | None = None
_items_returned_histogram: Histogram | None = None


def is_enabled() -> bool:
    """True once init() succeeds; False before init or after shutdown()."""
    return _provider is not None


def init(service_name: str) -> None:
    """Wire up the REST-metrics MeterProvider and instruments.

    No-op when already initialized. Does not set the global meter provider —
    uses its own reference to avoid clobbering the chief-only inventory
    telemetry provider.

    :param service_name: Bare service name (e.g. ``"api"``); ``"mlrun-"`` is
                         prepended for the OTel resource name.
    """
    global _provider, _meter
    global \
        _duration_histogram, \
        _request_size_histogram, \
        _response_size_histogram, \
        _items_returned_histogram

    if _provider is not None:
        mlrun.utils.logger.warning(
            "REST metrics telemetry already initialized; skipping re-init"
        )
        return

    _provider = framework.utils.telemetry.otel.build_metric_provider(
        service_name=f"mlrun-{service_name}"
    )
    _meter = _provider.get_meter("mlrun.rest_metrics")
    # Explicit-bucket histograms: universally supported. Exponential histograms
    # are a better fit for latency but require Prometheus native histogram
    # support, which isn't confirmed in all target deployments yet.
    _duration_histogram = _meter.create_histogram(
        name=_DURATION_INSTRUMENT_NAME,
        unit="ms",
        description="Duration of REST API request processing, in milliseconds.",
    )
    _request_size_histogram = _meter.create_histogram(
        name=_REQUEST_SIZE_INSTRUMENT_NAME,
        unit="KiBy",
        description="Size of the REST request body, in kibibytes.",
    )
    _response_size_histogram = _meter.create_histogram(
        name=_RESPONSE_SIZE_INSTRUMENT_NAME,
        unit="KiBy",
        description="Size of the REST response body, in kibibytes.",
    )
    _items_returned_histogram = _meter.create_histogram(
        name=_ITEMS_RETURNED_INSTRUMENT_NAME,
        unit="1",
        description="Number of objects returned by list REST calls.",
    )
    mlrun.utils.logger.info(
        "REST metrics telemetry instruments registered",
        service_name=service_name,
        otlp_endpoint=mlrun.mlconf.telemetry.otlp_endpoint,
        instruments=[
            _DURATION_INSTRUMENT_NAME,
            _REQUEST_SIZE_INSTRUMENT_NAME,
            _RESPONSE_SIZE_INSTRUMENT_NAME,
            _ITEMS_RETURNED_INSTRUMENT_NAME,
        ],
    )


def shutdown(timeout_millis: int = 2000) -> None:
    """Flush pending samples and tear down the MeterProvider.

    No-op when telemetry was never initialized. The short default timeout
    bounds how long an unreachable collector can stall pod termination.
    """
    global _provider, _meter
    global \
        _duration_histogram, \
        _request_size_histogram, \
        _response_size_histogram, \
        _items_returned_histogram
    if _provider is None:
        return
    try:
        _provider.shutdown(timeout_millis=timeout_millis)
        mlrun.utils.logger.info("REST metrics telemetry flushed and torn down")
    except Exception as exc:
        mlrun.utils.logger.warning(
            "REST metrics telemetry shutdown failed",
            error=mlrun.errors.err_to_str(exc),
        )
    finally:
        _provider = None
        _meter = None
        _duration_histogram = None
        _request_size_histogram = None
        _response_size_histogram = None
        _items_returned_histogram = None


def record_duration(
    duration_ms: float,
    method: str,
    status_code: int,
    resource: str,
    project: str,
) -> None:
    """Record request-processing duration (ms). No-op when telemetry is off.

    :param duration_ms:  Processing time in milliseconds.
    :param method:       HTTP method or ``"LIST"`` for collection GETs.
    :param status_code:  HTTP response status code.
    :param resource:     Object type (e.g. ``"functions"``), or ``""`` if none.
    :param project:      Project name for project-scoped routes, else ``""``.
    """
    _record(
        _duration_histogram.record if _duration_histogram is not None else None,
        duration_ms,
        method=method,
        status_code=status_code,
        resource=resource,
        project=project,
    )


def record_request_size(
    size_kib: float,
    method: str,
    status_code: int,
    resource: str,
    project: str,
) -> None:
    """Record request body size (KiB). No-op when telemetry is off.

    :param size_kib: Request body size in kibibytes.
    """
    _record(
        _request_size_histogram.record if _request_size_histogram is not None else None,
        size_kib,
        method=method,
        status_code=status_code,
        resource=resource,
        project=project,
    )


def record_response_size(
    size_kib: float,
    method: str,
    status_code: int,
    resource: str,
    project: str,
) -> None:
    """Record response body size (KiB). No-op when telemetry is off.

    :param size_kib: Response body size in kibibytes.
    """
    _record(
        _response_size_histogram.record
        if _response_size_histogram is not None
        else None,
        size_kib,
        method=method,
        status_code=status_code,
        resource=resource,
        project=project,
    )


def record_items_returned(
    item_count: int,
    status_code: int,
    resource: str,
    project: str,
) -> None:
    """Record item count for a list call. No-op when telemetry is off.

    Only call this after a list-shaped response body was successfully parsed.
    ``method`` is omitted from attributes — it's always ``"LIST"`` for this
    instrument, so it adds no query value.

    :param item_count: Number of objects returned by the call.
    """
    _record(
        _items_returned_histogram.record
        if _items_returned_histogram is not None
        else None,
        item_count,
        method="",
        status_code=status_code,
        resource=resource,
        project=project,
    )


def _record(
    record_fn: collections.abc.Callable[..., None] | None,
    value: float,
    *,
    method: str,
    status_code: int,
    resource: str,
    project: str,
) -> None:
    """Build shared attributes and call record_fn; no-op when record_fn is None.

    Callers pass ``method=""`` to omit it from attributes entirely — used by
    ``record_items_returned`` where method is always ``"LIST"`` and adds no
    query value.
    """
    if record_fn is None:
        return
    attributes = {
        "system_id": mlrun.mlconf.system_id or "",
        "status_code": status_code,
        "resource": resource,
        "project": project,
    }
    if method:
        attributes["method"] = method
    record_fn(value, attributes=attributes)
