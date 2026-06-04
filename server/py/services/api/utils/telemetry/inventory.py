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

"""Periodic-snapshot OTel telemetry for MLRun system inventory.

Chief-only. Re-anchored to DB truth on every project-summaries cache refresh,
so the values are immune to counter resets, pod restarts, and Prometheus
retention windows. Exported via synchronous Gauge instruments — one per
logical metric in ``_METRIC_NAMES``, all tagged with ``system_id`` plus any
per-call attributes (e.g. ``project``).

The OTLP exporter ticks at ``cache_interval`` * ``export_interval_multiplier``
(default 10 * 60 s = 10 min), aligned with the cache refresh cadence so each
export reflects a freshly anchored snapshot rather than an interpolated
between-refresh value.

Call sites:
  - ``init()`` from the chief's FastAPI startup hook.
  - ``shutdown()`` from the chief's FastAPI shutdown hook — flushes the
    final snapshot before pod termination.
  - ``set_count(metric, value, **attributes)`` from
    ``crud.Projects.refresh_project_resources_counters_cache()`` after the
    cache refresh, once per (metric, attribute-set) tuple.
"""

from typing import TypeVar

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Meter, Synchronous
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

import mlrun
import mlrun.errors
import mlrun.utils
import mlrun.utils.telemetry

# The concrete synchronous Gauge class is private (`_Gauge`) in
# opentelemetry-api 1.42, so bind the TypeVar to its public base instead.
Gauge = TypeVar("Gauge", bound=Synchronous)

# One Gauge per logical metric. Names mirror the fields already populated by
# ``refresh_project_resources_counters_cache`` — adding a metric here without
# a corresponding ``set_count`` call from the cache hook simply yields an
# empty series.
_METRIC_NAMES = (
    "mlrun_projects",
    "mlrun_artifacts",
    "mlrun_feature_sets",
    "mlrun_functions",
    "mlrun_schedules",
    "mlrun_schedules_pending",
    "mlrun_runs",
    "mlrun_workflows",
    "mlrun_pipeline_executions",
    "mlrun_alerts",
    "mlrun_alert_activations",
    "mlrun_model_endpoints",
    "mlrun_model_monitoring_functions",
)

_provider: MeterProvider | None = None
_meter: Meter | None = None
_gauges: dict[str, Gauge] = {}


def is_enabled() -> bool:
    """Whether the OTel SDK was successfully initialized.

    Single source of truth for callers that want to skip the cost of building
    an emission payload when telemetry is off — ``init()`` is the only place
    that flips this true, and it stays true until ``shutdown()`` resets state.
    """
    return _provider is not None


def init() -> None:
    """Wire up the inventory MeterProvider and Gauge instruments.

    The OTel SDK is loaded at import time; this function only stands up the
    chief-side exporter and registers gauges. No-op when telemetry is
    disabled or no OTLP endpoint is set; subsequent ``set_count`` calls then
    short-circuit on the empty gauge dict.

    Idempotent: a second call without an intervening ``shutdown()`` is a
    no-op, so a stray re-init (hot reload, test harness, double startup
    hook) doesn't orphan the previous MeterProvider's export thread + gRPC
    channel.
    """
    global _provider, _meter, _gauges

    if _provider is not None:
        mlrun.utils.logger.warning(
            "Telemetry inventory already initialized; skipping re-init"
        )
        return

    cfg = mlrun.mlconf.telemetry
    enabled = str(cfg.enabled).lower() == "true"
    if not enabled or not cfg.otlp_endpoint:
        mlrun.utils.logger.info(
            "Telemetry inventory disabled — gauges not registered",
            enabled=enabled,
            otlp_endpoint=cfg.otlp_endpoint or "<blank>",
        )
        return

    insecure = str(cfg.insecure).lower() == "true"
    # Gauges are re-set every cache cycle, so the exporter is aligned to that
    # cadence: export every Nth cycle (default N=10 → 10 minutes at the default
    # 60s cache_interval). Sub-1 config values are misconfigurations — clamp
    # to 1 and warn so the operator notices the override.
    raw_cache_interval = int(mlrun.mlconf.monitoring.projects.summaries.cache_interval)
    if raw_cache_interval < 1:
        mlrun.utils.logger.warning(
            "Telemetry inventory cache_interval < 1; clamping to 1",
            configured=raw_cache_interval,
        )
    cache_interval_seconds = max(1, raw_cache_interval)

    raw_multiplier = int(getattr(cfg.system_counters, "export_interval_multiplier", 10))
    if raw_multiplier < 1:
        mlrun.utils.logger.warning(
            "Telemetry inventory export_interval_multiplier < 1; clamping to 1",
            configured=raw_multiplier,
        )
    multiplier = max(1, raw_multiplier)
    export_interval_ms = multiplier * cache_interval_seconds * 1000

    exporter = OTLPMetricExporter(
        endpoint=cfg.otlp_endpoint,
        insecure=insecure,
        headers=mlrun.utils.telemetry.resolve_otlp_headers(),
    )
    reader = PeriodicExportingMetricReader(
        exporter, export_interval_millis=export_interval_ms
    )
    _provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(_provider)

    _meter = _provider.get_meter("mlrun.system")
    for name in _METRIC_NAMES:
        _gauges[name] = _meter.create_gauge(name=name)

    mlrun.utils.logger.info(
        "Telemetry inventory gauges registered",
        otlp_endpoint=cfg.otlp_endpoint,
        insecure=insecure,
        cache_interval_seconds=cache_interval_seconds,
        export_interval_multiplier=multiplier,
        export_interval_ms=export_interval_ms,
        metrics=list(_METRIC_NAMES),
    )


def shutdown(timeout_millis: int = 2000) -> None:
    """Flush any pending gauge values and tear down the MeterProvider.

    Called from the FastAPI shutdown hook so the final cache-refresh snapshot
    is exported before the chief pod terminates — without this, any gauge
    values set between the last exporter tick and pod termination are lost.
    No-op when telemetry was never initialized.

    The default ``timeout_millis`` is intentionally short (2s): the call sits
    in the async teardown path with nothing to run concurrently, so the
    timeout is the upper bound on how long an unreachable collector can stall
    pod termination. A healthy collector flushes in milliseconds.
    """
    global _provider, _meter, _gauges
    if _provider is None:
        return
    try:
        _provider.shutdown(timeout_millis=timeout_millis)
        mlrun.utils.logger.info("Telemetry inventory gauges flushed and torn down")
    except Exception as exc:
        mlrun.utils.logger.warning(
            "Telemetry inventory shutdown failed",
            error=mlrun.errors.err_to_str(exc),
        )
    finally:
        _provider = None
        _meter = None
        _gauges = {}


def set_count(metric: str, value: int, **attributes) -> None:
    """Set the current count for ``metric`` with the given attributes.

    No-op when the SDK was not initialized (telemetry disabled) or when
    ``metric`` is not in ``_METRIC_NAMES``. ``system_id`` is injected from
    ``mlrun.mlconf`` on every call so live config changes are picked up.
    """
    gauge = _gauges.get(metric)
    if gauge is None:
        return
    try:
        gauge.set(
            value,
            attributes={
                "system_id": mlrun.mlconf.system_id or "",
                **{k: (v or "") for k, v in attributes.items()},
            },
        )
    except Exception as exc:
        mlrun.utils.logger.warning(
            "Telemetry inventory emission failed",
            metric=metric,
            attributes=attributes,
            error=mlrun.errors.err_to_str(exc),
        )
