# Copyright 2023 Iguazio
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

import base64

import kubernetes.client
from kubernetes.client import ApiException

import mlrun.config
import mlrun.errors
import mlrun.k8s_utils
import mlrun.utils


def resolve_otlp_headers() -> dict[str, str]:
    """Resolve OTLP auth headers for telemetry exporters.

    Reads the K8s secret named in ``mlconf.telemetry.headers_secret_name`` directly
    via the in-cluster K8s API (no MLRun API call). Each key in the secret becomes
    one HTTP header (e.g. ``Authorization``, ``X-Scope-OrgID``).

    Works from both the API server and function pods. Returns an empty dict when
    no secret is configured, the caller is not running inside a K8s cluster, the
    namespace cannot be resolved, or the secret is missing/unreadable — telemetry
    must never break its caller.

    :returns: Mapping of header name -> header value. Empty dict if unconfigured
              or unresolvable.

    Example
    -------
    Operator creates a K8s secret with one key per outgoing header::

        kubectl create secret generic mlrun-otel-headers \\
            --from-literal=Authorization='Bearer eyJhbGc...' \\
            --from-literal=X-Scope-OrgID='tenant-42'

    And points MLRun at it via env var or helm values::

        MLRUN_TELEMETRY__OTLP_ENDPOINT=https://otel.example.com:4317
        MLRUN_TELEMETRY__INSECURE=false
        MLRUN_TELEMETRY__HEADERS_SECRET_NAME=mlrun-otel-headers

    Either the API server (chief pod, ML-16 system counters) or a function pod
    (model monitoring app, ML-12344) then builds an OTLP exporter::

        import mlrun
        import mlrun.utils.telemetry
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )

        exporter = OTLPMetricExporter(
            endpoint=mlrun.mlconf.telemetry.otlp_endpoint,
            insecure=mlrun.mlconf.telemetry.insecure == "true",
            headers=mlrun.utils.telemetry.resolve_otlp_headers(),
        )

    Rotation: ``kubectl edit secret mlrun-otel-headers`` is picked up the next
    time ``resolve_otlp_headers()`` is called (no pod restart required) — call
    it once per export cycle, not once per process.
    """
    secret_name = mlrun.mlconf.telemetry.headers_secret_name
    if not secret_name:
        return {}

    if not mlrun.k8s_utils.is_running_inside_kubernetes_cluster():
        return {}

    namespace = mlrun.mlconf.namespace
    if not namespace:
        mlrun.utils.logger.warning(
            "Cannot resolve OTLP telemetry headers — mlconf.namespace is unset",
            secret_name=secret_name,
        )
        return {}

    try:
        secret = kubernetes.client.CoreV1Api().read_namespaced_secret(
            name=secret_name, namespace=namespace
        )
    except ApiException as exc:
        mlrun.utils.logger.warning(
            "Failed to read OTLP telemetry headers secret",
            secret_name=secret_name,
            namespace=namespace,
            body=mlrun.errors.err_to_str(exc.body),
        )
        return {}

    headers = {
        key: base64.b64decode(value).decode("utf-8")
        for key, value in (secret.data or {}).items()
    }
    if headers:
        mlrun.utils.logger.debug(
            "Resolved OTLP telemetry headers",
            secret_name=secret_name,
            namespace=namespace,
            header_keys=sorted(headers.keys()),
        )
    return headers
