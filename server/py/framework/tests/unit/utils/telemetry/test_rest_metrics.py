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
import http
import unittest.mock

import pytest

import mlrun

import framework.utils.telemetry.rest_metrics as telemetry_rest_metrics

_ALL_INSTRUMENT_ATTRS = (
    "_duration_histogram",
    "_request_size_histogram",
    "_response_size_histogram",
    "_items_returned_histogram",
)


@pytest.fixture
def reset_state() -> collections.abc.Iterator[None]:
    """Wipe module-level state before and after each test.

    init() / shutdown() mutate module globals; tests must not leak.
    """
    telemetry_rest_metrics._provider = None
    telemetry_rest_metrics._meter = None
    for attr in _ALL_INSTRUMENT_ATTRS:
        setattr(telemetry_rest_metrics, attr, None)
    yield
    if telemetry_rest_metrics._provider is not None:
        telemetry_rest_metrics.shutdown(timeout_millis=100)
    telemetry_rest_metrics._provider = None
    telemetry_rest_metrics._meter = None
    for attr in _ALL_INSTRUMENT_ATTRS:
        setattr(telemetry_rest_metrics, attr, None)


@pytest.fixture
def telemetry_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure mlconf for a successful init(): master + sub flags + endpoint."""
    monkeypatch.setattr(mlrun.mlconf.telemetry, "enabled", True)
    monkeypatch.setattr(mlrun.mlconf.telemetry.rest_metrics, "enabled", True)
    monkeypatch.setattr(mlrun.mlconf.telemetry, "otlp_endpoint", "localhost:4317")
    monkeypatch.setattr(mlrun.mlconf.telemetry, "insecure", True)
    monkeypatch.setattr(mlrun.mlconf.telemetry, "headers_secret_name", "")


def test_is_enabled_false_before_init(reset_state: None) -> None:
    assert telemetry_rest_metrics.is_enabled() is False


def test_init_registers_all_instruments_when_enabled(
    reset_state: None, telemetry_enabled: None
) -> None:
    telemetry_rest_metrics.init(service_name="api")

    assert telemetry_rest_metrics.is_enabled() is True
    for attr in _ALL_INSTRUMENT_ATTRS:
        assert getattr(telemetry_rest_metrics, attr) is not None


def test_init_is_idempotent(
    reset_state: None, telemetry_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    warning_mock = unittest.mock.MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "warning", warning_mock)

    telemetry_rest_metrics.init(service_name="api")
    first_provider = telemetry_rest_metrics._provider
    assert first_provider is not None

    telemetry_rest_metrics.init(service_name="api")

    assert telemetry_rest_metrics._provider is first_provider
    warning_mock.assert_called_once()
    assert "already initialized" in warning_mock.call_args.args[0]


def test_shutdown_noop_when_uninitialized(reset_state: None) -> None:
    telemetry_rest_metrics.shutdown()
    assert telemetry_rest_metrics._provider is None


def test_shutdown_clears_module_state(reset_state: None) -> None:
    fake_provider = unittest.mock.MagicMock()
    telemetry_rest_metrics._provider = fake_provider
    telemetry_rest_metrics._meter = unittest.mock.MagicMock()
    for attr in _ALL_INSTRUMENT_ATTRS:
        setattr(telemetry_rest_metrics, attr, unittest.mock.MagicMock())

    telemetry_rest_metrics.shutdown(timeout_millis=1234)

    fake_provider.shutdown.assert_called_once_with(timeout_millis=1234)
    assert telemetry_rest_metrics._provider is None
    assert telemetry_rest_metrics._meter is None
    for attr in _ALL_INSTRUMENT_ATTRS:
        assert getattr(telemetry_rest_metrics, attr) is None


def test_record_duration_noop_when_uninitialized(reset_state: None) -> None:
    # Must not raise when the histogram was never created (telemetry off).
    telemetry_rest_metrics.record_duration(
        0.5,
        method="LIST",
        status_code=200,
        resource="runs",
        project="p",
    )
    assert telemetry_rest_metrics._duration_histogram is None


def test_record_duration_records_with_system_id_and_attributes(
    reset_state: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mlrun.mlconf, "system_id", "sys-xyz")
    histogram = unittest.mock.MagicMock()
    telemetry_rest_metrics._duration_histogram = histogram

    telemetry_rest_metrics.record_duration(
        0.25,
        method=http.HTTPMethod.POST,
        status_code=201,
        resource="functions",
        project="proj-a",
    )

    histogram.record.assert_called_once_with(
        0.25,
        attributes={
            "system_id": "sys-xyz",
            "method": http.HTTPMethod.POST,
            "status_code": 201,
            "resource": "functions",
            "project": "proj-a",
        },
    )


def test_record_duration_omits_method_key_when_empty(reset_state: None) -> None:
    """method is omitted entirely (not attached as "") when passed as ""."""
    histogram = unittest.mock.MagicMock()
    telemetry_rest_metrics._duration_histogram = histogram

    telemetry_rest_metrics.record_duration(
        0.1,
        method="",
        status_code=204,
        resource="runs",
        project="proj-a",
    )

    attributes = histogram.record.call_args.kwargs["attributes"]
    assert "method" not in attributes


def test_record_request_size_noop_when_uninitialized(reset_state: None) -> None:
    telemetry_rest_metrics.record_request_size(
        1.5,
        method=http.HTTPMethod.POST,
        status_code=200,
        resource="runs",
        project="p",
    )
    assert telemetry_rest_metrics._request_size_histogram is None


def test_record_request_size_records_with_attributes(reset_state: None) -> None:
    histogram = unittest.mock.MagicMock()
    telemetry_rest_metrics._request_size_histogram = histogram

    telemetry_rest_metrics.record_request_size(
        2.5,
        method=http.HTTPMethod.POST,
        status_code=201,
        resource="runs",
        project="proj-a",
    )

    histogram.record.assert_called_once_with(
        2.5,
        attributes={
            "system_id": "",
            "method": http.HTTPMethod.POST,
            "status_code": 201,
            "resource": "runs",
            "project": "proj-a",
        },
    )


def test_record_response_size_records_with_attributes(reset_state: None) -> None:
    histogram = unittest.mock.MagicMock()
    telemetry_rest_metrics._response_size_histogram = histogram

    telemetry_rest_metrics.record_response_size(
        12.75,
        method="LIST",
        status_code=200,
        resource="artifacts",
        project="proj-a",
    )

    histogram.record.assert_called_once_with(
        12.75,
        attributes={
            "system_id": "",
            "method": "LIST",
            "status_code": 200,
            "resource": "artifacts",
            "project": "proj-a",
        },
    )


def test_record_items_returned_noop_when_uninitialized(reset_state: None) -> None:
    telemetry_rest_metrics.record_items_returned(
        5, status_code=200, resource="runs", project="p"
    )
    assert telemetry_rest_metrics._items_returned_histogram is None


def test_record_items_returned_records_without_method(
    reset_state: None,
) -> None:
    """method is always "LIST" for this metric — it never varies, so it's not
    attached as an attribute.
    """
    histogram = unittest.mock.MagicMock()
    telemetry_rest_metrics._items_returned_histogram = histogram

    telemetry_rest_metrics.record_items_returned(
        7,
        status_code=200,
        resource="runs",
        project="proj-a",
    )

    histogram.record.assert_called_once_with(
        7,
        attributes={
            "system_id": "",
            "status_code": 200,
            "resource": "runs",
            "project": "proj-a",
        },
    )
