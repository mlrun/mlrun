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

import json
import os
import socket
from typing import Literal

import storey

import mlrun
import mlrun.errors
import mlrun.utils.telemetry

logger = mlrun.utils.logger


# TODO move to storey
def _warmup_endpoint(endpoint: str, timeout: float = 5.0) -> None:
    host, _, port = endpoint.rpartition(":")
    if not host or not port.isdigit():
        return
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            pass
    except OSError as exc:
        logger.debug(
            "OTel endpoint warmup probe failed",
            endpoint=endpoint,
            error=str(exc),
        )


# Sources for the OTLP headers passed to the underlying gRPC channel.
_HEADERS_SOURCE_FILE = "file"
_HEADERS_SOURCE_PROJECT_SECRET = "project_secret"
_HEADERS_SOURCE_NONE = "none"
_HEADERS_SOURCES = (
    _HEADERS_SOURCE_FILE,
    _HEADERS_SOURCE_PROJECT_SECRET,
    _HEADERS_SOURCE_NONE,
)


class OTelMetricsExporter(storey.OTelMetricsExporter):
    """MLRun serving graph step that exports OTel metrics as a side-effect.

    Inherits from ``storey.OTelMetricsExporter`` (a pass-through ``Flow`` step
    that forwards each event downstream after recording the metric) and layers
    MLRun-aware defaults on top:

    - ``endpoint`` defaults to ``mlrun.mlconf.telemetry.otlp_endpoint`` — set
      by the operator on the API server and delivered to the SDK via
      ``/client-spec``. Pass explicitly to route metrics to a different OTLP
      receiver.
    - ``insecure`` defaults to ``mlrun.mlconf.telemetry.insecure``.
    - Headers come from one of three sources controlled by ``headers_source``:

        * ``"file"`` (default): read from the kubelet-mounted secret at
          ``mlrun.common.constants.MLRUN_TELEMETRY_OTLP_HEADERS_PATH``. The
          server-side runtime injector mounts the secret when the function's
          ``runtime.spec.mount_otlp_secret=True``. One file per header —
          filename = header name, contents = header value. Used by MLRun's
          internal Model Monitoring applications.
        * ``"project_secret"``: read from a single project secret whose value
          is a JSON dict of ``{header_name: header_value}``. Use this when
          app authors want to manage their own OTel auth headers without
          touching the operator's secret.
        * ``"none"``: no headers. Suitable for in-cluster collectors that
          don't require authentication.

    Example::

        flow = function.set_topology("flow", engine="async")
        flow.to(name="my_app", class_name="MyApp").to(
            class_name="mlrun.serving.OTelMetricsExporter",
            # endpoint, insecure default from mlconf.telemetry
            headers_source="file",
        )

    .. warning::

       When ``headers_source`` is ``"file"`` or ``"project_secret"``, headers
       are resolved **eagerly** in ``__init__``. Always add the step via the
       ``class_name="mlrun.serving.OTelMetricsExporter"`` form (above) so
       construction is deferred to function-pod startup, where the secret
       mount (or project-secret env) actually exists. Instantiating the class
       directly on the SDK side — e.g.
       ``flow.to(OTelMetricsExporter(headers_source="file"))`` — runs the
       resolver against a missing mount, silently returns an empty headers
       dict, and bakes that empty dict into the serialized graph.

    The OTLP endpoint is resolved at construction time; if neither passed
    nor present in ``mlconf.telemetry``, construction raises
    ``MLRunRuntimeError``. Call ``mlrun.get_run_db()`` (or
    ``mlrun.get_or_create_project(...)``) first in dev contexts so the SDK
    has synced ``/client-spec``.

    :param endpoint: OTLP gRPC endpoint URL (e.g. ``"otel-collector.iguazio
                     .svc.cluster.local:4317"``). Defaults to
                     ``mlrun.mlconf.telemetry.otlp_endpoint``.
    :param insecure: Use a plaintext (non-TLS) gRPC channel. Defaults to
                     ``mlrun.mlconf.telemetry.insecure``.
    :param headers_source: One of ``"file"``, ``"project_secret"``, ``"none"``.
                           See class docstring above.
    :param project_secret_key: Required when ``headers_source="project_secret"``.
                               The secret's value must be a JSON object whose
                               keys are header names and values are header
                               values.

    All remaining keyword arguments are forwarded to
    ``storey.OTelMetricsExporter`` unchanged (``flush_mode``,
    ``export_interval_millis``, ``instrument_type``, ``metric_name_field``,
    ``value_field``, ``attribute_fields``, ``metrics_field``, etc.).
    """

    def __init__(
        self,
        endpoint: str | None = None,
        insecure: bool | None = None,
        headers_source: Literal["file", "project_secret", "none"] = "file",
        project_secret_key: str | None = None,
        export_interval_millis: int | None = None,
        **kwargs,
    ):
        if headers_source not in _HEADERS_SOURCES:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown headers_source {headers_source!r}; "
                f"use one of {_HEADERS_SOURCES}."
            )
        if headers_source == _HEADERS_SOURCE_PROJECT_SECRET and not project_secret_key:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "project_secret_key is required when headers_source='project_secret'."
            )

        resolved_endpoint = endpoint or mlrun.mlconf.telemetry.otlp_endpoint
        if not resolved_endpoint:
            raise mlrun.errors.MLRunRuntimeError(
                "OTLP endpoint unresolved: pass endpoint=..., or ensure the "
                "operator has set mlconf.telemetry.otlp_endpoint and the "
                "/client-spec sync has run (e.g. call mlrun.get_run_db() "
                "or mlrun.get_or_create_project() first)."
            )

        resolved_insecure = (
            insecure if insecure is not None else mlrun.mlconf.telemetry.insecure
        )

        mm_interval_s = mlrun.mlconf.telemetry.model_monitoring.interval
        export_interval_millis = export_interval_millis or (
            int(mm_interval_s) * 1000 if mm_interval_s is not None else None
        )
        if export_interval_millis is None:
            kwargs["flush_mode"] = "immediate"
        else:
            kwargs["export_interval_millis"] = export_interval_millis

        super().__init__(
            endpoint=resolved_endpoint,
            headers=self._resolve_headers(headers_source, project_secret_key),
            insecure=bool(resolved_insecure),
            **kwargs,
        )

        # todo move to storey
        if os.environ.get("NUCLIO_FUNCTION_NAME"):
            _warmup_endpoint(resolved_endpoint)

        # Stash mlrun-specific config for introspection / serialization.
        self._mlrun_headers_source = headers_source
        self._mlrun_project_secret_key = project_secret_key

    @staticmethod
    def _resolve_headers(source: str, project_secret_key: str | None) -> dict[str, str]:
        if source == _HEADERS_SOURCE_NONE:
            return {}
        if source == _HEADERS_SOURCE_FILE:
            return mlrun.utils.telemetry.resolve_otlp_headers()
        # project_secret
        raw = mlrun.get_secret_or_env(project_secret_key)
        if not raw:
            return {}
        try:
            headers = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Project secret {project_secret_key!r} must contain a JSON dict "
                f"of {{header_name: header_value}}; failed to parse."
            ) from exc
        if not isinstance(headers, dict):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Project secret {project_secret_key!r} must decode to a JSON dict; "
                f"got {type(headers).__name__}."
            )
        return {str(k): str(v) for k, v in headers.items()}
