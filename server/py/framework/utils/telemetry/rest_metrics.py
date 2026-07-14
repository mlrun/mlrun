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

"""Per-REST-call processing-time OTel telemetry for MLRun API-bearing services.

Runs on every replica (API chief + workers and the alerts service) — unlike the
chief-only inventory telemetry. Request durations are recorded to a single
Histogram instrument tagged with ``system_id`` plus bounded ``method`` and
``status_code`` attributes, exported over OTLP and aggregatable in Grafana.

The whole feature sits behind the master ``telemetry.enabled`` kill-switch and
the ``telemetry.rest_metrics.enabled`` sub-flag: when either is off (or no OTLP
endpoint is set) ``init()`` is a no-op, no provider is created, and
``record_duration()`` short-circuits — true zero-cost off.

Call sites:
  - ``init()`` from the shared service startup path (all replicas).
  - ``shutdown()`` from the shared service teardown path — flushes pending
    samples before pod termination.
  - ``record_duration(...)`` from ``RestMetricsMiddleware`` per REST call.
"""

from opentelemetry.metrics import Histogram, Meter
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
_INSTRUMENT_NAME = "mlrun_rest_request_duration_milliseconds"

_provider: MeterProvider | None = None
_meter: Meter | None = None
_histogram: Histogram | None = None


def is_enabled() -> bool:
    """Whether the OTel SDK was successfully initialized.

    ``init()`` is the only place that flips this true; it stays true until
    ``shutdown()`` resets state.
    """
    return _provider is not None


def init(service_name: str) -> None:
    """Wire up the REST-metrics MeterProvider and Histogram instrument.

    No-op when already initialized — a stray re-init (hot reload, double
    startup hook) won't orphan the previous export thread + gRPC channel.

    Does NOT call ``metrics.set_meter_provider`` — the meter is taken from
    this module's own provider reference so it never clobbers the global
    provider that the chief-only inventory telemetry claims.

    :param service_name: Bare service name (e.g. ``"api"``); ``"mlrun-"`` is
                         prepended internally to form the OTel resource name.
    """
    global _provider, _meter, _histogram

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
    _histogram = _meter.create_histogram(
        name=_INSTRUMENT_NAME,
        unit="ms",
        description="Duration of REST API request processing, in milliseconds.",
    )

    mlrun.utils.logger.info(
        "REST metrics telemetry histogram registered",
        service_name=service_name,
        otlp_endpoint=mlrun.mlconf.telemetry.otlp_endpoint,
        instrument=_INSTRUMENT_NAME,
    )


def shutdown(timeout_millis: int = 2000) -> None:
    """Flush any pending samples and tear down the MeterProvider.

    Called from the service teardown path so samples recorded between the last
    exporter tick and pod termination are exported. No-op when telemetry was
    never initialized. The short default ``timeout_millis`` bounds how long an
    unreachable collector can stall pod termination.
    """
    global _provider, _meter, _histogram
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
        _histogram = None


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
    picked up. Emission is wrapped so a misbehaving instrument can never fail
    the request that triggered it.

    :param duration_ms:  Processing time in milliseconds.
    :param method:       HTTP method (e.g. ``GET``).
    :param status_code:  HTTP response status code.
    :param resource:     Object type the route operates on (e.g. ``functions``),
                         or "" when none applies.
    :param project:      Project name for project-scoped routes, else "".
    """
    if _histogram is None:
        return
    try:
        _histogram.record(
            duration_ms,
            attributes={
                "system_id": mlrun.mlconf.system_id or "",
                "method": method,
                "status_code": status_code,
                "resource": resource,
                "project": project,
            },
        )
    except Exception as exc:
        mlrun.utils.logger.warning(
            "REST metrics telemetry emission failed",
            method=method,
            status_code=status_code,
            resource=resource,
            project=project,
            error=mlrun.errors.err_to_str(exc),
        )
