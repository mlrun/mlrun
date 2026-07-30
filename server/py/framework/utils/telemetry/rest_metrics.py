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

Runs on every replica (API chief + workers and the alerts service) — unlike the
chief-only inventory telemetry. Each call is recorded to a handful of
instruments (processing-time, request/response size, and — for list calls —
items-returned histograms), all tagged with ``system_id`` plus bounded
``method``/``status_code``/``resource``/``project`` attributes, exported over
OTLP and aggregatable in Grafana. ``method`` is the real HTTP method, except a
collection-returning GET is reported as the synthetic ``"LIST"`` value instead
of ``"GET"`` — see ``framework.middlewares.rest_metrics.parse_method`` — so
list calls are distinguishable from single-object gets without a separate
label.

The whole feature sits behind the master ``telemetry.enabled`` kill-switch and
the ``telemetry.rest_metrics.enabled`` sub-flag: when either is off (or no OTLP
endpoint is set) ``init()`` is a no-op, no provider is created, and the
``record_*()`` functions short-circuit — true zero-cost off.

Sampling (``telemetry.rest_metrics.sample_rate``) trades off overhead against
completeness: routine calls are recorded with probability ``sample_rate``,
while failed/slow/large calls are always recorded so those signals are never
missed. ``should_sample()`` is the single decision point; the middleware calls
it once per request and skips all ``record_*()`` calls when it returns False.
Grafana counts need a ``1 / sample_rate`` compensation factor whenever
sampling is enabled.

The configured ``sample_rate`` is also exported as its own gauge
(``mlrun_rest_metrics_sample_rate_ratio``) from the API chief only — one
point per system per export tick. Workers and the alerts service share the
same config value, so emitting identical ``{system_id: "X", value: 0.5}``
series from every replica adds no information. Callers control this via the
``emit_sample_rate_gauge`` argument to ``init()``. The gauge is
an ``ObservableGauge`` rather than a plain ``Gauge``: the SDK itself
re-invokes its callback on every export tick, so it keeps showing up in every
collection for as long as the process runs, with no need to re-``set()`` it
from request-handling code (a plain ``Gauge``'s last value is only exported
once and then dropped until ``set()`` is called again — see
``_LastValueAggregation.collect()`` in the OTel SDK).

Call sites:
  - ``init()`` from the shared service startup path (all replicas).
  - ``shutdown()`` from the shared service teardown path — flushes pending
    samples before pod termination.
  - ``should_sample(...)`` and the ``record_*(...)`` functions from
    ``RestMetricsMiddleware`` per REST call.
"""

import collections.abc
import random

from opentelemetry.metrics import (
    CallbackOptions,
    Histogram,
    Meter,
    ObservableGauge,
    Observation,
)
from opentelemetry.sdk.metrics import MeterProvider

import mlrun
import mlrun.errors
import mlrun.utils

import framework.utils.telemetry.otel

# Milliseconds (not the Prometheus/OTel base unit of seconds) on purpose: the
# SDK's default histogram buckets (5, 10, 25, ... 10000) map onto real request
# latencies in ms, giving useful resolution out of the box. Seconds would put
# almost every request in the first bucket unless we also defined a custom View
# with sub-second boundaries. The ``_milliseconds`` suffix + ``ms`` unit signal
# the non-base unit to consumers.
_DURATION_INSTRUMENT_NAME = "mlrun_rest_request_duration_milliseconds"
# "_kibibytes", not "_kilobytes": the OTel<->Prometheus exporter expands the
# unit="KiBy" below to the literal word "kibibytes" and only skips
# re-appending it as a suffix if the name already ends with that *exact*
# word — same mechanism that makes "ms" above work with "_milliseconds".
# "_kilobytes" doesn't match "kibibytes" textually, so it doesn't count,
# and the name+unit combination must agree on binary (1024) vs decimal
# (1000) for the value to be honest either way. See the OTel<->Prometheus
# metric-metadata docs:
# https://opentelemetry.io/docs/specs/otel/compatibility/prometheus_and_openmetrics/#metric-metadata
_REQUEST_SIZE_INSTRUMENT_NAME = "mlrun_rest_request_size_kibibytes"
_RESPONSE_SIZE_INSTRUMENT_NAME = "mlrun_rest_response_size_kibibytes"
_ITEMS_RETURNED_INSTRUMENT_NAME = "mlrun_rest_response_num_items"
# "_ratio", matching unit="1"'s OTel<->Prometheus expansion (same
# name/unit-agreement rule as the "_kibibytes" histograms above) —
# verified empirically: without it, the exporter appended its own
# "_ratio" suffix on top, giving "..._sample_rate_ratio" anyway, just
# with a name that doesn't self-document the unit.
_SAMPLE_RATE_INSTRUMENT_NAME = "mlrun_rest_metrics_sample_rate_ratio"

# Always-keep carve-outs for should_sample() (unlike sample_rate, these aren't
# operator-tunable — they encode what "slow" and "large" mean for this feature,
# not a deployment-specific tradeoff).
_SLOW_THRESHOLD_SECONDS = 10
_LARGE_RESPONSE_KIB = 100

_provider: MeterProvider | None = None
_meter: Meter | None = None
_duration_histogram: Histogram | None = None
_request_size_histogram: Histogram | None = None
_response_size_histogram: Histogram | None = None
_items_returned_histogram: Histogram | None = None
_sample_rate_gauge: ObservableGauge | None = None


def is_enabled() -> bool:
    """Whether the OTel SDK was successfully initialized.

    ``init()`` is the only place that flips this true; it stays true until
    ``shutdown()`` resets state.
    """
    return _provider is not None


def init(service_name: str, *, emit_sample_rate_gauge: bool = True) -> None:
    """Wire up the REST-metrics MeterProvider and Histogram instrument.

    No-op when already initialized — a stray re-init (hot reload, double
    startup hook) won't orphan the previous export thread + gRPC channel.

    Does NOT call ``metrics.set_meter_provider`` — the meter is taken from
    this module's own provider reference so it never clobbers the global
    provider that the chief-only inventory telemetry claims.

    :param service_name:          Bare service name (e.g. ``"api"``);
                                  ``"mlrun-"`` is prepended internally to form
                                  the OTel resource name.
    :param emit_sample_rate_gauge: When ``False``, the sample-rate gauge is not
                                   created. Pass ``False`` for every replica
                                   that is not the API chief — one point per
                                   system is enough.
    """
    global _provider, _meter
    global \
        _duration_histogram, \
        _request_size_histogram, \
        _response_size_histogram, \
        _items_returned_histogram, \
        _sample_rate_gauge

    if _provider is not None:
        mlrun.utils.logger.warning(
            "REST metrics telemetry already initialized; skipping re-init"
        )
        return

    _provider = framework.utils.telemetry.otel.build_metric_provider(
        service_name=f"mlrun-{service_name}"
    )
    _meter = _provider.get_meter("mlrun.rest_metrics")
    # Default (explicit-bucket) histogram aggregation — universally supported on
    # any Prometheus/Grafana. An exponential histogram would give auto-scaling,
    # tuning-free resolution and is the better fit for latency, but it maps to
    # Prometheus native histograms (feature-flagged + newer OTLP/Grafana support)
    # and would degrade silently on stacks without it. Revisit via a View once
    # native-histogram support is confirmed in the target deployment.
    _duration_histogram = _meter.create_histogram(
        name=_DURATION_INSTRUMENT_NAME,
        unit="ms",
        description="Duration of REST API request processing, in milliseconds.",
    )
    # unit="KiBy" must agree with the "_kibibytes" already in the name above
    # (see the comment on _REQUEST_SIZE_INSTRUMENT_NAME).
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
    instruments = [
        _DURATION_INSTRUMENT_NAME,
        _REQUEST_SIZE_INSTRUMENT_NAME,
        _RESPONSE_SIZE_INSTRUMENT_NAME,
        _ITEMS_RETURNED_INSTRUMENT_NAME,
    ]
    if emit_sample_rate_gauge:
        _sample_rate_gauge = _meter.create_observable_gauge(
            name=_SAMPLE_RATE_INSTRUMENT_NAME,
            callbacks=[_sample_rate_callback],
            unit="1",
            description=(
                "Configured REST-metrics sample rate, for compensating "
                "count-based queries on the other instruments."
            ),
        )
        instruments.append(_SAMPLE_RATE_INSTRUMENT_NAME)
    mlrun.utils.logger.info(
        "REST metrics telemetry instruments registered",
        service_name=service_name,
        otlp_endpoint=mlrun.mlconf.telemetry.otlp_endpoint,
        instruments=instruments,
    )


def shutdown(timeout_millis: int = 2000) -> None:
    """Flush any pending samples and tear down the MeterProvider.

    Called from the service teardown path so samples recorded between the last
    exporter tick and pod termination are exported. No-op when telemetry was
    never initialized. The short default ``timeout_millis`` bounds how long an
    unreachable collector can stall pod termination.
    """
    global _provider, _meter
    global \
        _duration_histogram, \
        _request_size_histogram, \
        _response_size_histogram, \
        _items_returned_histogram, \
        _sample_rate_gauge
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
        _sample_rate_gauge = None


def _sample_rate_callback(
    options: CallbackOptions,
) -> collections.abc.Iterable[Observation]:
    """Report the currently configured sample rate on every export tick.

    Reading live config here (rather than capturing the rate once at
    ``init()`` time) means a config change takes effect on the very next
    tick, with no re-init needed.
    """
    yield Observation(
        mlrun.mlconf.telemetry.rest_metrics.sample_rate,
        attributes={"system_id": mlrun.mlconf.system_id or ""},
    )


def should_sample(
    *,
    status_code: int,
    elapsed_seconds: float,
    response_size_kib: float,
) -> bool:
    """Decide whether a call's metrics should be recorded.

    Failed, slow, or large calls are always kept regardless of the configured
    rate — those are exactly the calls an admin most needs visibility into, and
    sampling them out would hide the signal they exist to surface. Everything
    else is kept with probability ``telemetry.rest_metrics.sample_rate``.

    Called exactly once per call, after every instrument's value is known, so
    a call is either kept for all of them or dropped for all of them — never a
    mix.

    :param status_code:       HTTP response status code.
    :param elapsed_seconds:   Processing time in seconds.
    :param response_size_kib: Response body size in kibibytes.
    :return: Whether the call's metrics should be recorded.
    """
    if (
        status_code >= 300
        or elapsed_seconds > _SLOW_THRESHOLD_SECONDS
        or response_size_kib > _LARGE_RESPONSE_KIB
    ):
        return True
    return random.random() < mlrun.mlconf.telemetry.rest_metrics.sample_rate


def record_duration(
    duration_ms: float,
    method: str,
    status_code: int,
    resource: str,
    project: str,
) -> None:
    """Record a single request-processing duration to the histogram.

    No-op when the SDK was not initialized (telemetry disabled). ``system_id``
    is injected from ``mlrun.mlconf`` on every call so live config changes are
    picked up.

    :param duration_ms:  Processing time in milliseconds.
    :param method:       HTTP method (e.g. ``GET``), or the synthetic
                         ``"LIST"`` value for a collection-returning GET.
    :param status_code:  HTTP response status code.
    :param resource:     Object type the route operates on (e.g. ``functions``),
                         or "" when none applies.
    :param project:      Project name for project-scoped routes, else "".
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
    """Record a single request body size, in kibibytes, to the histogram.

    See ``record_duration`` for the shared no-op behavior and parameter
    meanings.

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
    """Record a single response body size, in kibibytes, to the histogram.

    See ``record_duration`` for the shared no-op behavior and parameter
    meanings.

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
    """Record the number of objects a list call returned to the items histogram.

    A histogram (not a counter) on purpose: like duration and size, this is a
    per-call value, not a running total — a histogram preserves the per-call
    distribution (e.g. p95 list size) as well as the sum/count an aggregate
    query would want, whereas a counter only ever gives the latter.

    Only meaningful for list calls — callers must only invoke this once the
    response body was successfully parsed and a list-shaped payload found.
    ``method`` is always ``"LIST"`` for every point on this metric, so unlike
    the other ``record_*()`` functions it's not accepted or attached as an
    attribute here — a label that never varies within a metric adds nothing to
    query it by.

    See ``record_duration`` for the shared no-op behavior and remaining
    parameter meanings.

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
    """Shared attributes-building + no-op logic.

    ``record_fn`` is the instrument's bound recording method — ``Histogram.record``
    or ``Counter.add`` — pre-selected by the caller (both take the value and an
    ``attributes`` mapping, so any instrument kind can share this helper). None
    when the instrument was never initialized (telemetry disabled).

    ``method`` is required here — every caller must make an explicit choice —
    but is omitted entirely (rather than attached as "") when passed as "": a
    label that never varies within a metric adds nothing to query it by.
    ``record_items_returned`` is the only caller that does this, since
    ``method`` never varies (always ``"LIST"``) for that specific instrument.
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
