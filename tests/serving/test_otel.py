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

import contextlib
import json
from unittest.mock import patch

import pytest
import storey

import mlrun
import mlrun.common.constants
import mlrun.errors
from mlrun.serving import OTelMetricsExporter

_TEST_ENDPOINT = "otel-collector.iguazio.svc.cluster.local:4317"


@contextlib.contextmanager
def _telemetry_config(
    *,
    endpoint: str | None = None,
    insecure: bool | None = None,
):
    """Temporarily set mlconf.telemetry.* for one test; restore after."""
    saved_endpoint = mlrun.mlconf.telemetry.otlp_endpoint
    saved_insecure = mlrun.mlconf.telemetry.insecure
    if endpoint is not None:
        mlrun.mlconf.telemetry.otlp_endpoint = endpoint
    if insecure is not None:
        mlrun.mlconf.telemetry.insecure = insecure
    try:
        yield
    finally:
        mlrun.mlconf.telemetry.otlp_endpoint = saved_endpoint
        mlrun.mlconf.telemetry.insecure = saved_insecure


def test_inherits_from_storey_exporter():
    """The MLRun step IS a storey.OTelMetricsExporter — graph machinery that
    type-checks against the storey base must still match."""
    assert issubclass(OTelMetricsExporter, storey.OTelMetricsExporter)


def test_endpoint_explicit_overrides_mlconf():
    with _telemetry_config(endpoint="https://mlconf-endpoint:4317"):
        step = OTelMetricsExporter(
            endpoint="https://explicit:4317",
            headers_source="none",
        )
    # storey stashes endpoint on _endpoint
    assert step._endpoint == "https://explicit:4317"


def test_endpoint_falls_back_to_mlconf():
    with _telemetry_config(endpoint=_TEST_ENDPOINT):
        step = OTelMetricsExporter(headers_source="none")
    assert step._endpoint == _TEST_ENDPOINT


def test_endpoint_unresolved_raises():
    with _telemetry_config(endpoint=""):
        with pytest.raises(
            mlrun.errors.MLRunRuntimeError, match="OTLP endpoint unresolved"
        ):
            OTelMetricsExporter(headers_source="none")


def test_insecure_explicit_overrides_mlconf():
    with _telemetry_config(endpoint=_TEST_ENDPOINT, insecure=False):
        step = OTelMetricsExporter(insecure=True, headers_source="none")
    assert step._insecure is True


def test_insecure_falls_back_to_mlconf():
    with _telemetry_config(endpoint=_TEST_ENDPOINT, insecure=False):
        step = OTelMetricsExporter(headers_source="none")
    assert step._insecure is False


def test_headers_source_none_yields_empty_headers():
    with _telemetry_config(endpoint=_TEST_ENDPOINT):
        step = OTelMetricsExporter(headers_source="none")
    assert step._headers == {}


def test_headers_source_file_reads_via_resolve_helper(tmp_path):
    """`headers_source="file"` delegates to
    `mlrun.utils.telemetry.resolve_otlp_headers()`, which reads
    filename->header-name, contents->header-value from
    `MLRUN_TELEMETRY_OTLP_HEADERS_PATH`. We mock the helper here to keep
    the test independent of the kubelet mount layout.
    """
    expected = {"Authorization": "Bearer abc123", "X-Scope-OrgID": "tenant-7"}
    with _telemetry_config(endpoint=_TEST_ENDPOINT):
        with patch(
            "mlrun.utils.telemetry.resolve_otlp_headers", return_value=expected
        ) as resolve_mock:
            step = OTelMetricsExporter(headers_source="file")
    resolve_mock.assert_called_once()
    assert step._headers == expected


def test_headers_source_project_secret_loads_json_dict():
    secret_blob = json.dumps(
        {"Authorization": "Bearer xyz", "X-Scope-OrgID": "tenant-9"}
    )
    with _telemetry_config(endpoint=_TEST_ENDPOINT):
        with patch(
            "mlrun.get_secret_or_env", return_value=secret_blob
        ) as get_secret_mock:
            step = OTelMetricsExporter(
                headers_source="project_secret",
                project_secret_key="OTLP_HEADERS",
            )
    get_secret_mock.assert_called_once_with("OTLP_HEADERS")
    assert step._headers == {
        "Authorization": "Bearer xyz",
        "X-Scope-OrgID": "tenant-9",
    }


def test_headers_source_project_secret_requires_key():
    with _telemetry_config(endpoint=_TEST_ENDPOINT):
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="project_secret_key is required",
        ):
            OTelMetricsExporter(headers_source="project_secret")


def test_headers_source_project_secret_invalid_json_raises():
    with _telemetry_config(endpoint=_TEST_ENDPOINT):
        with patch("mlrun.get_secret_or_env", return_value="not-json"):
            with pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError, match="JSON dict"
            ):
                OTelMetricsExporter(
                    headers_source="project_secret",
                    project_secret_key="OTLP_HEADERS",
                )


def test_headers_source_project_secret_non_dict_raises():
    """JSON that decodes to a list (or anything else) is rejected — the
    contract is a flat `{header_name: header_value}` mapping.
    """
    with _telemetry_config(endpoint=_TEST_ENDPOINT):
        with patch("mlrun.get_secret_or_env", return_value=json.dumps(["a", "b"])):
            with pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError, match="JSON dict"
            ):
                OTelMetricsExporter(
                    headers_source="project_secret",
                    project_secret_key="OTLP_HEADERS",
                )


def test_headers_source_project_secret_missing_returns_empty():
    """Project secret unset (returns None) → empty headers dict, no auth."""
    with _telemetry_config(endpoint=_TEST_ENDPOINT):
        with patch("mlrun.get_secret_or_env", return_value=None):
            step = OTelMetricsExporter(
                headers_source="project_secret",
                project_secret_key="OTLP_HEADERS",
            )
    assert step._headers == {}


def test_unknown_headers_source_raises():
    with _telemetry_config(endpoint=_TEST_ENDPOINT):
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError, match="Unknown headers_source"
        ):
            OTelMetricsExporter(headers_source="bogus")


def test_storey_kwargs_passthrough():
    """Storey-only knobs (flush_mode, instrument_type, etc.) reach the base."""
    with _telemetry_config(endpoint=_TEST_ENDPOINT):
        step = OTelMetricsExporter(
            headers_source="none",
            flush_mode="immediate",
            instrument_type="counter",
            max_instruments=42,
        )
    assert step._flush_mode == "immediate"
    assert step._instrument_type == "counter"
    assert step._max_instruments == 42


def test_mlrun_introspection_attrs_set():
    """We stash the original headers_source / project_secret_key on the
    instance so callers (and the graph serializer, if needed later) can
    recover the original intent rather than just the resolved dict.
    """
    with _telemetry_config(endpoint=_TEST_ENDPOINT):
        step = OTelMetricsExporter(
            headers_source="project_secret",
            project_secret_key="OTLP_HEADERS",
        )
    assert step._mlrun_headers_source == "project_secret"
    assert step._mlrun_project_secret_key == "OTLP_HEADERS"
