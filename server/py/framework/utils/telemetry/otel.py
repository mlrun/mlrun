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

"""Shared OTel metrics bootstrap for server-side telemetry features.

Every metrics feature (inventory gauges, REST-metrics histogram, ...) stands up
the same three pieces identically — an OTLP gRPC exporter, a periodic reader,
and a Resource carrying service identity. ``build_metric_provider`` assembles
them so each feature only owns its own enable gate and instrument registration.
"""

import os
import socket

from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

import mlrun
import mlrun.utils.telemetry


def build_metric_provider(
    service_name: str,
    export_interval_millis: int | None = None,
) -> MeterProvider:
    """Build an OTLP-exporting ``MeterProvider`` from the telemetry config.

    Assembles the OTLP gRPC exporter (endpoint / insecure / auth headers from
    ``mlrun.mlconf.telemetry``), a ``PeriodicExportingMetricReader``, and a
    ``Resource`` carrying service identity, and returns the provider. The caller
    owns the enable gate, whether to set it as the global provider, and which
    instruments to register on it.

    ``service.name`` → Prometheus ``job`` label, ``service.instance.id`` →
    ``instance`` label. Pod name comes from the MLRUN_POD_NAME downward-API env
    var, falling back to the hostname (which K8s sets to the pod name).

    :param service_name:           OTel ``service.name`` for this provider.
    :param export_interval_millis: Reader export interval in ms; omit for the
                                   SDK default.
    :returns: A configured ``MeterProvider`` (not registered as the global one).
    """
    cfg = mlrun.mlconf.telemetry
    exporter = OTLPMetricExporter(
        endpoint=cfg.otlp_endpoint,
        insecure=cfg.insecure,
        headers=mlrun.utils.telemetry.resolve_otlp_headers(),
    )
    reader_kwargs = {}
    if export_interval_millis is not None:
        reader_kwargs["export_interval_millis"] = export_interval_millis
    reader = PeriodicExportingMetricReader(exporter, **reader_kwargs)
    pod_name = os.getenv("MLRUN_POD_NAME") or socket.gethostname()
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.instance.id": pod_name,
        }
    )
    return MeterProvider(metric_readers=[reader], resource=resource)


def build_log_provider(service_name: str) -> LoggerProvider:
    """Build an OTLP-exporting ``LoggerProvider`` from the telemetry config.

    Mirrors ``build_metric_provider`` — same endpoint, insecure flag, and
    ``Resource`` carrying service identity. The caller owns the enable gate
    and which loggers to acquire from the provider.

    :param service_name: OTel ``service.name`` for this provider.
    :returns: A configured ``LoggerProvider`` (not registered as the global one).
    """
    cfg = mlrun.mlconf.telemetry
    exporter = OTLPLogExporter(
        endpoint=cfg.otlp_endpoint,
        insecure=cfg.insecure,
        headers=mlrun.utils.telemetry.resolve_otlp_headers(),
    )
    pod_name = os.getenv("MLRUN_POD_NAME") or socket.gethostname()
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.instance.id": pod_name,
        }
    )
    provider = LoggerProvider(resource=resource)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    return provider
