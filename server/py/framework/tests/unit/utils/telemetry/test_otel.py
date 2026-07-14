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

import collections.abc

import pytest

import mlrun

import framework.utils.telemetry.otel as telemetry_otel


@pytest.fixture
def telemetry_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoint/insecure/headers needed to construct the exporter."""
    monkeypatch.setattr(mlrun.mlconf.telemetry, "otlp_endpoint", "localhost:4317")
    monkeypatch.setattr(mlrun.mlconf.telemetry, "insecure", True)
    monkeypatch.setattr(mlrun.mlconf.telemetry, "headers_secret_name", "")


def _spy_meter_provider(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    real_cls = telemetry_otel.MeterProvider

    def _spy(*args, resource=None, **kwargs):
        captured["resource"] = resource
        return real_cls(*args, resource=resource, **kwargs)

    monkeypatch.setattr(telemetry_otel, "MeterProvider", _spy)


def _spy_reader(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    real_cls = telemetry_otel.PeriodicExportingMetricReader

    def _spy(exporter, **kwargs):
        captured["reader_kwargs"] = kwargs
        return real_cls(exporter, **kwargs)

    monkeypatch.setattr(telemetry_otel, "PeriodicExportingMetricReader", _spy)


@pytest.fixture(autouse=True)
def _no_thread_leak() -> collections.abc.Iterator[list]:
    """Shut down any provider a test builds so its export thread doesn't leak."""
    built: list = []
    yield built
    for provider in built:
        provider.shutdown(timeout_millis=100)


def test_build_metric_provider_sets_service_name_and_pod_name(
    telemetry_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    _no_thread_leak: list,
) -> None:
    monkeypatch.setenv("MLRUN_POD_NAME", "mlrun-api-7c9-xyz")
    captured: dict = {}
    _spy_meter_provider(monkeypatch, captured)

    provider = telemetry_otel.build_metric_provider("mlrun-api")
    _no_thread_leak.append(provider)

    attributes = captured["resource"].attributes
    assert attributes["service.name"] == "mlrun-api"
    assert attributes["service.instance.id"] == "mlrun-api-7c9-xyz"


def test_build_metric_provider_pod_name_falls_back_to_hostname(
    telemetry_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    _no_thread_leak: list,
) -> None:
    monkeypatch.delenv("MLRUN_POD_NAME", raising=False)
    monkeypatch.setattr(telemetry_otel.socket, "gethostname", lambda: "host-fallback")
    captured: dict = {}
    _spy_meter_provider(monkeypatch, captured)

    provider = telemetry_otel.build_metric_provider("mlrun-alerts")
    _no_thread_leak.append(provider)

    assert captured["resource"].attributes["service.instance.id"] == "host-fallback"


def test_build_metric_provider_uses_sdk_default_interval_when_omitted(
    telemetry_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    _no_thread_leak: list,
) -> None:
    captured: dict = {}
    _spy_reader(monkeypatch, captured)

    provider = telemetry_otel.build_metric_provider("mlrun-api")
    _no_thread_leak.append(provider)

    # omitted → not forwarded, so the SDK's own default interval applies
    assert "export_interval_millis" not in captured["reader_kwargs"]


def test_build_metric_provider_passes_custom_interval(
    telemetry_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    _no_thread_leak: list,
) -> None:
    captured: dict = {}
    _spy_reader(monkeypatch, captured)

    provider = telemetry_otel.build_metric_provider(
        "mlrun-api", export_interval_millis=5000
    )
    _no_thread_leak.append(provider)

    assert captured["reader_kwargs"]["export_interval_millis"] == 5000


def test_build_metric_provider_exporter_uses_endpoint_and_insecure(
    telemetry_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    _no_thread_leak: list,
) -> None:
    captured: dict = {}
    real_cls = telemetry_otel.OTLPMetricExporter

    def _spy(**kwargs):
        captured.update(kwargs)
        return real_cls(**kwargs)

    monkeypatch.setattr(telemetry_otel, "OTLPMetricExporter", _spy)

    provider = telemetry_otel.build_metric_provider("mlrun-api")
    _no_thread_leak.append(provider)

    assert captured["endpoint"] == "localhost:4317"
    assert captured["insecure"] is True
