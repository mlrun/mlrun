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
import unittest.mock

import pytest

import mlrun

import services.api.utils.telemetry.inventory as telemetry_inventory


@pytest.fixture
def reset_inventory_state() -> collections.abc.Iterator[None]:
    """Wipe inventory module-level state before and after each test.

    init() / shutdown() mutate module globals; tests must not leak across
    each other or into the rest of the suite.
    """
    telemetry_inventory._provider = None
    telemetry_inventory._meter = None
    telemetry_inventory._gauges = {}
    yield
    if telemetry_inventory._provider is not None:
        telemetry_inventory.shutdown(timeout_millis=100)
    telemetry_inventory._provider = None
    telemetry_inventory._meter = None
    telemetry_inventory._gauges = {}


@pytest.fixture
def telemetry_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure mlconf for a successful init(): enabled + non-blank endpoint."""
    monkeypatch.setattr(mlrun.mlconf.telemetry, "enabled", True)
    monkeypatch.setattr(mlrun.mlconf.telemetry, "otlp_endpoint", "localhost:4317")
    monkeypatch.setattr(mlrun.mlconf.telemetry, "insecure", True)
    monkeypatch.setattr(mlrun.mlconf.telemetry, "headers_secret_name", "")


def test_is_enabled_false_before_init(reset_inventory_state: None) -> None:
    """is_enabled() must reflect a never-initialized SDK as disabled."""
    assert telemetry_inventory.is_enabled() is False


def test_is_enabled_true_after_successful_init(
    reset_inventory_state: None,
    telemetry_enabled: None,
) -> None:
    """is_enabled() flips to True once init() wires up a real provider."""
    telemetry_inventory.init()
    assert telemetry_inventory.is_enabled() is True


def test_is_enabled_false_after_disabled_init(
    reset_inventory_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disabled-config init() leaves is_enabled() False."""
    monkeypatch.setattr(mlrun.mlconf.telemetry, "enabled", False)
    telemetry_inventory.init()
    assert telemetry_inventory.is_enabled() is False


def test_init_noop_when_telemetry_disabled(
    reset_inventory_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enabled=False → SDK not initialized, no gauges registered."""
    monkeypatch.setattr(mlrun.mlconf.telemetry, "enabled", False)
    monkeypatch.setattr(mlrun.mlconf.telemetry, "otlp_endpoint", "localhost:4317")

    telemetry_inventory.init()

    assert telemetry_inventory._provider is None
    assert telemetry_inventory._meter is None
    assert telemetry_inventory._gauges == {}


def test_init_noop_when_endpoint_missing(
    reset_inventory_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank otlp_endpoint disables telemetry regardless of `enabled`."""
    monkeypatch.setattr(mlrun.mlconf.telemetry, "enabled", True)
    monkeypatch.setattr(mlrun.mlconf.telemetry, "otlp_endpoint", "")

    telemetry_inventory.init()

    assert telemetry_inventory._provider is None
    assert telemetry_inventory._gauges == {}


def test_init_registers_gauge_for_every_metric_name(
    reset_inventory_state: None,
    telemetry_enabled: None,
) -> None:
    """Each name in `_METRIC_NAMES` gets a Gauge instrument after init()."""
    telemetry_inventory.init()

    assert telemetry_inventory._provider is not None
    assert telemetry_inventory._meter is not None
    assert set(telemetry_inventory._gauges.keys()) == set(
        telemetry_inventory._METRIC_NAMES
    )


def test_init_export_interval_is_cache_interval_times_multiplier(
    reset_inventory_state: None,
    telemetry_enabled: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """export_interval_ms == cache_interval_s * multiplier * 1000."""
    monkeypatch.setattr(
        mlrun.mlconf.monitoring.projects.summaries, "cache_interval", 30
    )
    monkeypatch.setattr(
        mlrun.mlconf.telemetry.system_counters, "export_interval_multiplier", 4
    )

    captured: dict = {}
    real_reader_cls = telemetry_inventory.PeriodicExportingMetricReader

    def _spy(exporter, export_interval_millis, **kwargs):
        captured["export_interval_millis"] = export_interval_millis
        return real_reader_cls(
            exporter, export_interval_millis=export_interval_millis, **kwargs
        )

    monkeypatch.setattr(telemetry_inventory, "PeriodicExportingMetricReader", _spy)

    telemetry_inventory.init()

    assert captured["export_interval_millis"] == 30 * 4 * 1000


@pytest.mark.parametrize(
    "cache_interval,multiplier,expected_ms,expected_warnings",
    [
        (0, 10, 1 * 10 * 1000, ["cache_interval"]),  # cache_interval clamps to 1
        (60, 0, 60 * 1 * 1000, ["export_interval_multiplier"]),  # multiplier clamps
        (-5, -3, 1 * 1 * 1000, ["cache_interval", "export_interval_multiplier"]),
    ],
)
def test_init_clamps_invalid_interval_config(
    reset_inventory_state: None,
    telemetry_enabled: None,
    monkeypatch: pytest.MonkeyPatch,
    cache_interval: int,
    multiplier: int,
    expected_ms: int,
    expected_warnings: list[str],
) -> None:
    """Non-positive cache_interval or multiplier values must clamp to 1 + warn."""
    monkeypatch.setattr(
        mlrun.mlconf.monitoring.projects.summaries, "cache_interval", cache_interval
    )
    monkeypatch.setattr(
        mlrun.mlconf.telemetry.system_counters,
        "export_interval_multiplier",
        multiplier,
    )

    captured: dict = {}
    real_reader_cls = telemetry_inventory.PeriodicExportingMetricReader

    def _spy(exporter, export_interval_millis, **kwargs):
        captured["export_interval_millis"] = export_interval_millis
        return real_reader_cls(
            exporter, export_interval_millis=export_interval_millis, **kwargs
        )

    monkeypatch.setattr(telemetry_inventory, "PeriodicExportingMetricReader", _spy)
    warning_mock = unittest.mock.MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "warning", warning_mock)

    telemetry_inventory.init()

    assert captured["export_interval_millis"] == expected_ms
    warning_messages = [call.args[0] for call in warning_mock.call_args_list]
    for needle in expected_warnings:
        assert any(needle in msg for msg in warning_messages), (
            f"missing clamp warning for {needle}; got: {warning_messages}"
        )
    assert len(warning_messages) == len(expected_warnings)


def test_init_is_idempotent(
    reset_inventory_state: None,
    telemetry_enabled: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second init() without an intervening shutdown() is a no-op + warning.

    Prevents orphaning the previous MeterProvider's export thread / gRPC
    channel on a stray re-init (hot reload, test harness, double startup hook).
    """
    warning_mock = unittest.mock.MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "warning", warning_mock)

    telemetry_inventory.init()
    first_provider = telemetry_inventory._provider
    first_gauges = telemetry_inventory._gauges
    assert first_provider is not None

    telemetry_inventory.init()

    assert telemetry_inventory._provider is first_provider
    assert telemetry_inventory._gauges is first_gauges
    warning_mock.assert_called_once()
    assert "already initialized" in warning_mock.call_args.args[0]


def test_shutdown_noop_when_uninitialized(reset_inventory_state: None) -> None:
    """shutdown() before init() must not raise."""
    telemetry_inventory.shutdown()

    assert telemetry_inventory._provider is None


def test_shutdown_clears_module_state(reset_inventory_state: None) -> None:
    """A successful shutdown resets provider/meter/gauges to their empty defaults."""
    fake_provider = unittest.mock.MagicMock()
    telemetry_inventory._provider = fake_provider
    telemetry_inventory._meter = unittest.mock.MagicMock()
    telemetry_inventory._gauges = {"mlrun_projects": unittest.mock.MagicMock()}

    telemetry_inventory.shutdown(timeout_millis=1234)

    fake_provider.shutdown.assert_called_once_with(timeout_millis=1234)
    assert telemetry_inventory._provider is None
    assert telemetry_inventory._meter is None
    assert telemetry_inventory._gauges == {}


def test_shutdown_swallows_provider_exceptions(
    reset_inventory_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing provider.shutdown() must not propagate, and state must still reset."""
    fake_provider = unittest.mock.MagicMock()
    fake_provider.shutdown.side_effect = RuntimeError("collector unreachable")
    telemetry_inventory._provider = fake_provider
    telemetry_inventory._meter = unittest.mock.MagicMock()
    telemetry_inventory._gauges = {"mlrun_projects": unittest.mock.MagicMock()}
    warning_mock = unittest.mock.MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "warning", warning_mock)

    telemetry_inventory.shutdown()

    assert telemetry_inventory._provider is None
    assert telemetry_inventory._gauges == {}
    warning_mock.assert_called_once()
    assert "shutdown failed" in warning_mock.call_args.args[0]


def test_set_count_noop_when_sdk_uninitialized(reset_inventory_state: None) -> None:
    """set_count() before init() (empty _gauges) is silently skipped."""
    telemetry_inventory.set_count("mlrun_projects", 5, project="p")
    # No assertion needed beyond "didn't raise" — the gauge dict is empty.
    assert telemetry_inventory._gauges == {}


def test_set_count_noop_when_metric_name_unknown(reset_inventory_state: None) -> None:
    """A name not in _gauges is dropped — no exception, no spurious emission."""
    known_gauge = unittest.mock.MagicMock()
    telemetry_inventory._gauges = {"mlrun_projects": known_gauge}

    telemetry_inventory.set_count("not_a_real_metric", 5, project="p")

    known_gauge.set.assert_not_called()


def test_set_count_injects_system_id_and_forwards_attributes(
    reset_inventory_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The current mlconf.system_id is added to every emission alongside user kwargs."""
    monkeypatch.setattr(mlrun.mlconf, "system_id", "sys-xyz")
    gauge = unittest.mock.MagicMock()
    telemetry_inventory._gauges = {"mlrun_artifacts": gauge}

    telemetry_inventory.set_count("mlrun_artifacts", 7, project="proj-a", kind="model")

    gauge.set.assert_called_once_with(
        7,
        attributes={"system_id": "sys-xyz", "project": "proj-a", "kind": "model"},
    )


def test_set_count_falls_back_to_empty_string_for_missing_values(
    reset_inventory_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None or "" attribute values + missing system_id are normalized to "".

    OTLP attribute values must be non-None primitives; None would break export.
    """
    monkeypatch.setattr(mlrun.mlconf, "system_id", "")
    gauge = unittest.mock.MagicMock()
    telemetry_inventory._gauges = {"mlrun_projects": gauge}

    telemetry_inventory.set_count("mlrun_projects", 0, project=None, kind="")

    gauge.set.assert_called_once_with(
        0,
        attributes={"system_id": "", "project": "", "kind": ""},
    )


def test_set_count_swallows_gauge_exceptions(
    reset_inventory_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misbehaving gauge.set() must not propagate to the cache-refresh hook."""
    gauge = unittest.mock.MagicMock()
    gauge.set.side_effect = RuntimeError("instrument broken")
    telemetry_inventory._gauges = {"mlrun_projects": gauge}
    warning_mock = unittest.mock.MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "warning", warning_mock)

    telemetry_inventory.set_count("mlrun_projects", 1, project="p")

    gauge.set.assert_called_once()
    warning_mock.assert_called_once()
    assert "emission failed" in warning_mock.call_args.args[0]
