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

import framework.utils.telemetry.rest_records as telemetry_rest_records


@pytest.fixture
def reset_state() -> collections.abc.Iterator[None]:
    """Wipe module-level state before and after each test."""
    telemetry_rest_records._provider = None
    telemetry_rest_records._otel_logger = None
    yield
    if telemetry_rest_records._provider is not None:
        telemetry_rest_records.shutdown()
    telemetry_rest_records._provider = None
    telemetry_rest_records._otel_logger = None


@pytest.fixture
def telemetry_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlrun.mlconf.telemetry, "enabled", True)
    monkeypatch.setattr(mlrun.mlconf.telemetry.rest_metrics, "enabled", True)
    monkeypatch.setattr(mlrun.mlconf.telemetry, "otlp_endpoint", "localhost:4317")
    monkeypatch.setattr(mlrun.mlconf.telemetry, "insecure", True)
    monkeypatch.setattr(mlrun.mlconf.telemetry, "headers_secret_name", "")


def test_is_enabled_false_before_init(reset_state: None) -> None:
    assert telemetry_rest_records.is_enabled() is False


def test_init_sets_provider_and_logger(
    reset_state: None, telemetry_enabled: None
) -> None:
    telemetry_rest_records.init(service_name="api")

    assert telemetry_rest_records.is_enabled() is True
    assert telemetry_rest_records._otel_logger is not None


def test_init_is_idempotent(
    reset_state: None, telemetry_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    warning_mock = unittest.mock.MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "warning", warning_mock)

    telemetry_rest_records.init(service_name="api")
    first_provider = telemetry_rest_records._provider

    telemetry_rest_records.init(service_name="api")

    assert telemetry_rest_records._provider is first_provider
    warning_mock.assert_called_once()
    assert "already initialized" in warning_mock.call_args.args[0]


def test_shutdown_noop_when_uninitialized(reset_state: None) -> None:
    telemetry_rest_records.shutdown()
    assert telemetry_rest_records._provider is None


def test_shutdown_clears_module_state(reset_state: None) -> None:
    fake_provider = unittest.mock.MagicMock()
    telemetry_rest_records._provider = fake_provider
    telemetry_rest_records._otel_logger = unittest.mock.MagicMock()

    telemetry_rest_records.shutdown()

    fake_provider.shutdown.assert_called_once()
    assert telemetry_rest_records._provider is None
    assert telemetry_rest_records._otel_logger is None


class TestShouldSampleRecord:
    @pytest.fixture(autouse=True)
    def _configure_thresholds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mlrun.mlconf.telemetry.rest_metrics, "sample_rate", 1.0)
        monkeypatch.setattr(telemetry_rest_records, "_SLOW_THRESHOLD_SECONDS", 10)
        monkeypatch.setattr(telemetry_rest_records, "_LARGE_RESPONSE_KIB", 100)

    def test_full_sample_rate_always_keeps_routine_calls(self) -> None:
        assert (
            telemetry_rest_records.should_sample_record(
                status_code=200, elapsed_seconds=0.1, response_size_kib=1
            )
            is True
        )

    def test_zero_sample_rate_drops_routine_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mlrun.mlconf.telemetry.rest_metrics, "sample_rate", 0.0)

        assert (
            telemetry_rest_records.should_sample_record(
                status_code=200, elapsed_seconds=0.1, response_size_kib=1
            )
            is False
        )

    def test_zero_sample_rate_still_keeps_failed_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mlrun.mlconf.telemetry.rest_metrics, "sample_rate", 0.0)

        assert (
            telemetry_rest_records.should_sample_record(
                status_code=500, elapsed_seconds=0.1, response_size_kib=1
            )
            is True
        )

    def test_zero_sample_rate_still_keeps_slow_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mlrun.mlconf.telemetry.rest_metrics, "sample_rate", 0.0)

        assert (
            telemetry_rest_records.should_sample_record(
                status_code=200, elapsed_seconds=11, response_size_kib=1
            )
            is True
        )

    def test_zero_sample_rate_still_keeps_large_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mlrun.mlconf.telemetry.rest_metrics, "sample_rate", 0.0)

        assert (
            telemetry_rest_records.should_sample_record(
                status_code=200, elapsed_seconds=0.1, response_size_kib=101
            )
            is True
        )

    def test_redirect_status_codes_are_treated_as_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mlrun.mlconf.telemetry.rest_metrics, "sample_rate", 0.0)

        assert (
            telemetry_rest_records.should_sample_record(
                status_code=302, elapsed_seconds=0.1, response_size_kib=1
            )
            is True
        )

    def test_sample_rate_respects_random_draw(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mlrun.mlconf.telemetry.rest_metrics, "sample_rate", 0.5)
        monkeypatch.setattr(telemetry_rest_records.random, "random", lambda: 0.4)
        assert (
            telemetry_rest_records.should_sample_record(
                status_code=200, elapsed_seconds=0.1, response_size_kib=1
            )
            is True
        )

        monkeypatch.setattr(telemetry_rest_records.random, "random", lambda: 0.6)
        assert (
            telemetry_rest_records.should_sample_record(
                status_code=200, elapsed_seconds=0.1, response_size_kib=1
            )
            is False
        )


def test_emit_record_noop_when_uninitialized(reset_state: None) -> None:
    # Must not raise when the logger was never created (telemetry off).
    telemetry_rest_records.emit_record(
        path="/api/v1/projects/p/runs",
        query_string="",
        method="GET",
        status_code=200,
        duration_ms=50.0,
        request_size_bytes=0,
        response_size_bytes=100,
        resource="runs",
        project="p",
        client_ip="1.2.3.4",
        request_id="req-1",
        item_count=None,
    )
    assert telemetry_rest_records._otel_logger is None


def test_emit_record_calls_logger_emit(reset_state: None) -> None:
    mock_logger = unittest.mock.MagicMock()
    telemetry_rest_records._otel_logger = mock_logger

    telemetry_rest_records.emit_record(
        path="/api/v1/projects/proj/functions",
        query_string="name=fn",
        method="LIST",
        status_code=200,
        duration_ms=120.5,
        request_size_bytes=0,
        response_size_bytes=2048,
        resource="functions",
        project="proj",
        client_ip="10.0.0.1",
        request_id="abc",
        item_count=5,
    )

    mock_logger.emit.assert_called_once()
    kwargs = mock_logger.emit.call_args.kwargs
    assert kwargs["body"] == "LIST /api/v1/projects/proj/functions?name=fn 200"
    attrs = kwargs["attributes"]
    assert attrs["method"] == "LIST"
    assert attrs["status_code"] == 200
    assert attrs["resource"] == "functions"
    assert attrs["project"] == "proj"
    assert attrs["items_returned"] == 5
    assert attrs["url"] == "/api/v1/projects/proj/functions?name=fn"


def test_emit_record_omits_item_count_when_none(reset_state: None) -> None:
    mock_logger = unittest.mock.MagicMock()
    telemetry_rest_records._otel_logger = mock_logger

    telemetry_rest_records.emit_record(
        path="/api/v1/projects/p/runs/uid",
        query_string="",
        method="GET",
        status_code=200,
        duration_ms=10.0,
        request_size_bytes=0,
        response_size_bytes=512,
        resource="runs",
        project="p",
        client_ip="",
        request_id="",
        item_count=None,
    )

    attrs = mock_logger.emit.call_args.kwargs["attributes"]
    assert "items_returned" not in attrs


def test_emit_record_url_without_query_string(reset_state: None) -> None:
    mock_logger = unittest.mock.MagicMock()
    telemetry_rest_records._otel_logger = mock_logger

    telemetry_rest_records.emit_record(
        path="/api/v1/projects/p/runs/uid",
        query_string="",
        method="GET",
        status_code=200,
        duration_ms=10.0,
        request_size_bytes=0,
        response_size_bytes=0,
        resource="runs",
        project="p",
        client_ip="",
        request_id="",
        item_count=None,
    )

    kwargs = mock_logger.emit.call_args.kwargs
    assert kwargs["body"] == "GET /api/v1/projects/p/runs/uid 200"
    assert kwargs["attributes"]["url"] == "/api/v1/projects/p/runs/uid"
