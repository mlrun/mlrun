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

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kubernetes.client import ApiException

import mlrun.config
import mlrun.k8s_utils
import mlrun.utils.telemetry


@pytest.fixture
def mlconf_telemetry(monkeypatch):
    """Reset relevant mlconf keys before each test."""
    monkeypatch.setattr(
        mlrun.mlconf.telemetry, "headers_secret_name", "", raising=False
    )
    monkeypatch.setattr(mlrun.mlconf, "namespace", "mlrun-ns", raising=False)
    yield


@pytest.fixture
def in_cluster(monkeypatch):
    monkeypatch.setattr(
        mlrun.k8s_utils, "is_running_inside_kubernetes_cluster", lambda: True
    )


@pytest.fixture
def out_of_cluster(monkeypatch):
    monkeypatch.setattr(
        mlrun.k8s_utils, "is_running_inside_kubernetes_cluster", lambda: False
    )


def _mock_secret(data: dict[str, str]) -> SimpleNamespace:
    """Build a V1Secret-like object with already-base64-encoded values."""
    return SimpleNamespace(
        data={k: base64.b64encode(v.encode()).decode() for k, v in data.items()}
    )


def _patch_corev1(monkeypatch, mock_api):
    monkeypatch.setattr("kubernetes.client.CoreV1Api", lambda *args, **kwargs: mock_api)


def test_returns_empty_when_secret_name_unset(mlconf_telemetry, in_cluster):
    assert mlrun.utils.telemetry.resolve_otlp_headers() == {}


def test_returns_empty_when_not_in_cluster(mlconf_telemetry, out_of_cluster):
    mlrun.mlconf.telemetry.headers_secret_name = "mlrun-otel-headers"
    assert mlrun.utils.telemetry.resolve_otlp_headers() == {}


def test_reads_headers_from_secret(mlconf_telemetry, in_cluster, monkeypatch):
    mlrun.mlconf.telemetry.headers_secret_name = "mlrun-otel-headers"
    api = MagicMock()
    api.read_namespaced_secret.return_value = _mock_secret(
        {"Authorization": "Bearer token-xyz", "X-Scope-OrgID": "tenant-42"}
    )
    _patch_corev1(monkeypatch, api)

    headers = mlrun.utils.telemetry.resolve_otlp_headers()

    assert headers == {
        "Authorization": "Bearer token-xyz",
        "X-Scope-OrgID": "tenant-42",
    }
    api.read_namespaced_secret.assert_called_once_with(
        name="mlrun-otel-headers", namespace="mlrun-ns"
    )


@pytest.mark.parametrize("status,reason", [(404, "Not Found"), (403, "Forbidden")])
def test_returns_empty_on_api_exception(
    mlconf_telemetry, in_cluster, monkeypatch, status, reason
):
    mlrun.mlconf.telemetry.headers_secret_name = "secret-name"
    api = MagicMock()
    api.read_namespaced_secret.side_effect = ApiException(status=status, reason=reason)
    _patch_corev1(monkeypatch, api)

    assert mlrun.utils.telemetry.resolve_otlp_headers() == {}


def test_returns_empty_when_secret_data_is_none(
    mlconf_telemetry, in_cluster, monkeypatch
):
    mlrun.mlconf.telemetry.headers_secret_name = "empty-secret"
    api = MagicMock()
    api.read_namespaced_secret.return_value = SimpleNamespace(data=None)
    _patch_corev1(monkeypatch, api)

    assert mlrun.utils.telemetry.resolve_otlp_headers() == {}


def test_returns_empty_when_secret_data_is_empty_dict(
    mlconf_telemetry, in_cluster, monkeypatch
):
    mlrun.mlconf.telemetry.headers_secret_name = "empty-secret"
    api = MagicMock()
    api.read_namespaced_secret.return_value = SimpleNamespace(data={})
    _patch_corev1(monkeypatch, api)

    assert mlrun.utils.telemetry.resolve_otlp_headers() == {}


def test_returns_empty_when_namespace_unset(mlconf_telemetry, in_cluster, monkeypatch):
    mlrun.mlconf.telemetry.headers_secret_name = "mlrun-otel-headers"
    monkeypatch.setattr(mlrun.mlconf, "namespace", "", raising=False)

    api = MagicMock()
    _patch_corev1(monkeypatch, api)

    assert mlrun.utils.telemetry.resolve_otlp_headers() == {}
    api.read_namespaced_secret.assert_not_called()


def test_decodes_complex_header_values(mlconf_telemetry, in_cluster, monkeypatch):
    """Header values with spaces, JWT-style payloads, and non-ASCII bytes round-trip cleanly."""
    mlrun.mlconf.telemetry.headers_secret_name = "mlrun-otel-headers"
    payload = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig",
        "X-Multi-Word": "value with spaces and = signs",
        "X-UTF8": "héllo-wörld",
        "X-Empty": "",
    }
    api = MagicMock()
    api.read_namespaced_secret.return_value = _mock_secret(payload)
    _patch_corev1(monkeypatch, api)

    assert mlrun.utils.telemetry.resolve_otlp_headers() == payload


def test_logs_warning_when_namespace_unset(mlconf_telemetry, in_cluster, monkeypatch):
    mlrun.mlconf.telemetry.headers_secret_name = "mlrun-otel-headers"
    monkeypatch.setattr(mlrun.mlconf, "namespace", "", raising=False)
    warn_spy = MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "warning", warn_spy)

    mlrun.utils.telemetry.resolve_otlp_headers()

    warn_spy.assert_called_once()
    args, kwargs = warn_spy.call_args
    assert "mlconf.namespace is unset" in args[0]
    assert kwargs.get("secret_name") == "mlrun-otel-headers"


def test_logs_warning_on_api_exception(mlconf_telemetry, in_cluster, monkeypatch):
    mlrun.mlconf.telemetry.headers_secret_name = "missing-secret"
    api = MagicMock()
    api.read_namespaced_secret.side_effect = ApiException(
        status=404, reason="Not Found"
    )
    _patch_corev1(monkeypatch, api)
    warn_spy = MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "warning", warn_spy)

    mlrun.utils.telemetry.resolve_otlp_headers()

    warn_spy.assert_called_once()
    args, kwargs = warn_spy.call_args
    assert "Failed to read OTLP telemetry headers secret" in args[0]
    assert kwargs.get("secret_name") == "missing-secret"
    assert kwargs.get("namespace") == "mlrun-ns"
