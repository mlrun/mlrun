# Copyright 2024 Iguazio
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

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any, Optional, TypedDict
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import v3io.dataplane.kv
import v3io.dataplane.output
import v3io.dataplane.response
import v3io_frames

import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.utils.v3io_clients
from mlrun.common.schemas.model_monitoring.model_endpoints import (
    ModelEndpointMonitoringMetric,
    ModelEndpointMonitoringMetricNoData,
    ModelEndpointMonitoringResultValues,
    _MetricPoint,
)
from mlrun.model_monitoring.db.stores.v3io_kv.kv_store import KVStoreBase
from mlrun.model_monitoring.db.tsdb.v3io.stream_graph_steps import (
    _normalize_dict_for_v3io_frames,
)
from mlrun.model_monitoring.db.tsdb.v3io.v3io_connector import (
    V3IOTSDBConnector,
    _is_no_schema_error,
)


@pytest.fixture(params=["default-project"])
def store(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> KVStoreBase:
    monkeypatch.setenv("V3IO_ACCESS_KEY", "secret-value")
    store = KVStoreBase(project=request.param)
    return store


@pytest.mark.parametrize(
    ("store", "items", "expected_metrics"),
    [
        ("", [], []),
        (
            "default",
            [
                {
                    "__name": "some-app-name",
                    "some-metric-name1": "string-attrs1",
                    "some-metric-name2": "string-attrs2",
                }
            ],
            [
                ModelEndpointMonitoringMetric(
                    project="default",
                    app="some-app-name",
                    type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
                    name="some-metric-name1",
                    full_name="default.some-app-name.result.some-metric-name1",
                ),
                ModelEndpointMonitoringMetric(
                    project="default",
                    app="some-app-name",
                    type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
                    name="some-metric-name2",
                    full_name="default.some-app-name.result.some-metric-name2",
                ),
            ],
        ),
        (
            "project-1",
            [
                {
                    "__name": "app1",
                    "drift-res": "{'d':0.4}",
                }
            ],
            [
                ModelEndpointMonitoringMetric(
                    project="project-1",
                    app="app1",
                    type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
                    name="drift-res",
                    full_name="project-1.app1.result.drift-res",
                )
            ],
        ),
    ],
    indirect=["store"],
)
def test_extract_results_from_items(
    store: KVStoreBase,
    items: list[dict[str, str]],
    expected_metrics: list[ModelEndpointMonitoringMetric],
) -> None:
    assert store._extract_results_from_items(items) == expected_metrics


@pytest.mark.parametrize(
    ("store", "items", "expected_metrics"),
    [
        ("", [], []),
        (
            "default",
            [
                {
                    "__name": "histogram-data-drift.tvd_mean",
                    "application_name": "histogram-data-drift",
                    "metric_name": "tvd_mean",
                    "metric_value": 1.0,
                    "start_infer_time": "2024-05-15 17:25:28.000000+00:00",
                    "end_infer_time": "2024-05-15 17:26:28.000000+00:00",
                },
                {
                    "__name": "histogram-data-drift.kld_mean",
                    "application_name": "histogram-data-drift",
                    "metric_name": "kld_mean",
                    "metric_value": 15.7973460213887,
                    "start_infer_time": "2024-05-15 17:25:28.000000+00:00",
                    "end_infer_time": "2024-05-15 17:26:28.000000+00:00",
                },
                {
                    "__name": "histogram-data-drift.hellinger_mean",
                    "application_name": "histogram-data-drift",
                    "metric_name": "hellinger_mean",
                    "metric_value": 1.0,
                    "start_infer_time": "2024-05-15 17:25:28.000000+00:00",
                    "end_infer_time": "2024-05-15 17:26:28.000000+00:00",
                },
            ],
            [
                ModelEndpointMonitoringMetric(
                    project="default",
                    app="histogram-data-drift",
                    type=mm_constants.ModelEndpointMonitoringMetricType.METRIC,
                    name="tvd_mean",
                    full_name="default.histogram-data-drift.metric.tvd_mean",
                ),
                ModelEndpointMonitoringMetric(
                    project="default",
                    app="histogram-data-drift",
                    type=mm_constants.ModelEndpointMonitoringMetricType.METRIC,
                    name="kld_mean",
                    full_name="default.histogram-data-drift.metric.kld_mean",
                ),
                ModelEndpointMonitoringMetric(
                    project="default",
                    app="histogram-data-drift",
                    type=mm_constants.ModelEndpointMonitoringMetricType.METRIC,
                    name="hellinger_mean",
                    full_name="default.histogram-data-drift.metric.hellinger_mean",
                ),
            ],
        ),
    ],
    indirect=["store"],
)
def test_extract_metrics_from_items(
    store: KVStoreBase,
    items: list[dict[str, str]],
    expected_metrics: list[ModelEndpointMonitoringMetric],
) -> None:
    assert store._extract_metrics_from_items(items) == expected_metrics


@pytest.mark.parametrize(
    ("exc", "expected_is_no_schema"),
    [
        (v3io_frames.ReadError("Another read error"), False),
        (
            v3io_frames.ReadError(
                "can't query: failed to create adapter: No TSDB schema file found at "
                "'v3io-webapi:8081/users/pipelines/mm-serving-no-data/model-endpoints/predictions/'."
            ),
            True,
        ),
    ],
)
def test_is_no_schema_error(
    exc: v3io_frames.ReadError, expected_is_no_schema: bool
) -> None:
    assert _is_no_schema_error(exc) == expected_is_no_schema


@pytest.fixture
def kv_client_mock() -> v3io.dataplane.kv.Model:
    kv_mock = Mock(spec=v3io.dataplane.kv.Model)
    schema_file = Mock(all=Mock(return_value=False))
    kv_mock.new_cursor = Mock(return_value=schema_file)
    kv_mock.create_schema = Mock(return_value=Mock(status_code=HTTPStatus.OK))
    return kv_mock


@pytest.fixture
def mocked_client_store(
    store: KVStoreBase, kv_client_mock: v3io.dataplane.kv.Model
) -> KVStoreBase:
    store.client.kv = kv_client_mock
    return store


@pytest.fixture
def metric_event() -> dict[str, Any]:
    return {
        mm_constants.WriterEvent.ENDPOINT_ID: "ep-id",
        mm_constants.WriterEvent.APPLICATION_NAME: "some-app",
        mm_constants.MetricData.METRIC_NAME: "metric_1",
        mm_constants.MetricData.METRIC_VALUE: 0.345,
        mm_constants.WriterEvent.START_INFER_TIME: "2024-05-10T13:00:00.0+00:00",
        mm_constants.WriterEvent.END_INFER_TIME: "2024-05-10T14:00:00.0+00:00",
    }


def test_write_application_metric(
    mocked_client_store: KVStoreBase,
    kv_client_mock: v3io.dataplane.kv.Model,
    metric_event: dict[str, Any],
) -> None:
    mocked_client_store.write_application_event(
        event=metric_event, kind=mm_constants.WriterEventKind.METRIC
    )
    kv_client_mock.update.assert_called_once()
    kv_client_mock.create_schema.assert_called_once()


class TestGetModelEndpointMetrics:
    PROJECT = "demo-proj"
    ENDPOINT = "70450e1ef7cc9506d42369aeeb056eaaaa0bb8bd"
    CONTAINER = KVStoreBase.get_v3io_monitoring_apps_container(PROJECT)

    @dataclass
    class Entry:
        class DecodedBody(TypedDict):
            LastItemIncluded: str
            NextMarker: str

        decoded_body: DecodedBody
        items: list[dict[str, str]]

    _M0 = "r55ggss="
    _M1 = "reeedkdk2="
    _V0 = Entry(
        decoded_body=Entry.DecodedBody(LastItemIncluded="FALSE", NextMarker=_M1),
        items=[{"__name": "perf-app", "latency-sec": ""}],
    )
    _V1 = Entry(
        decoded_body=Entry.DecodedBody(LastItemIncluded="TRUE", NextMarker=_M0),
        items=[
            {"__name": "model-as-a-judge-app", "distance": ""},
        ],
    )
    METRICS = [
        ModelEndpointMonitoringMetric(
            project=PROJECT,
            app="perf-app",
            type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
            name="latency-sec",
            full_name="demo-proj.perf-app.result.latency-sec",
        ),
        ModelEndpointMonitoringMetric(
            project=PROJECT,
            app="model-as-a-judge-app",
            type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
            name="distance",
            full_name="demo-proj.model-as-a-judge-app.result.distance",
        ),
    ]

    MARKER_TO_ENTRY_MAP: dict[Optional[str], Entry] = {None: _V0, _M0: _V0, _M1: _V1}

    @classmethod
    def scan_mock(
        cls,
        container: str,
        table_path: str,
        marker: Optional[str] = None,
        filter_expression: Optional[str] = None,
    ) -> v3io.dataplane.response.Response:
        if (
            container != cls.CONTAINER
            or table_path != cls.ENDPOINT
            or marker not in cls.MARKER_TO_ENTRY_MAP
        ):
            raise v3io.dataplane.response.HttpResponseError(
                status_code=HTTPStatus.NOT_FOUND
            )
        entry = cls.MARKER_TO_ENTRY_MAP[marker]
        response = v3io.dataplane.response.Response(
            output="",
            status_code=HTTPStatus.OK,
            headers={},
            body=b"",
        )
        response._parsed_output = v3io.dataplane.output.GetItemsOutput(
            decoded_body=entry.decoded_body
        )
        response._parsed_output.items = entry.items
        return response

    @classmethod
    @pytest.fixture
    def store(cls, monkeypatch: pytest.MonkeyPatch) -> KVStoreBase:
        monkeypatch.setenv("V3IO_ACCESS_KEY", "secret-value")
        store = KVStoreBase(project=cls.PROJECT)
        monkeypatch.setattr(store.client.kv, "scan", cls.scan_mock)
        return store

    @staticmethod
    @pytest.fixture
    def response_error() -> v3io.dataplane.response.HttpResponseError:
        return v3io.dataplane.response.HttpResponseError(
            status_code=HTTPStatus.NETWORK_AUTHENTICATION_REQUIRED
        )

    @staticmethod
    @pytest.fixture
    def store_with_err(
        store: KVStoreBase, response_error: v3io.dataplane.response.HttpResponseError
    ) -> Iterator[KVStoreBase]:
        with patch.object(store.client.kv, "scan", Mock(side_effect=response_error)):
            yield store

    @classmethod
    @pytest.mark.parametrize(
        ("endpoint_id", "expected_metrics"),
        [("non-existent-ep", []), (ENDPOINT, METRICS)],
    )
    def test_metrics(
        cls,
        store: KVStoreBase,
        endpoint_id: str,
        expected_metrics: list[ModelEndpointMonitoringMetric],
    ) -> None:
        assert (
            store.get_model_endpoint_metrics(
                endpoint_id, type=mm_constants.ModelEndpointMonitoringMetricType.RESULT
            )
            == expected_metrics
        )

    @classmethod
    def test_response_error(cls, store_with_err: KVStoreBase) -> None:
        """Test that non 404 errors are not silenced"""
        with pytest.raises(v3io.dataplane.response.HttpResponseError):
            store_with_err.get_model_endpoint_metrics(
                cls.ENDPOINT, type=mm_constants.ModelEndpointMonitoringMetricType.RESULT
            )


@pytest.mark.parametrize(
    ("endpoint_id", "names", "table_path", "columns", "expected_query"),
    [
        (
            "ddw2lke",
            [],
            "app-results",
            None,
            "SELECT * FROM 'app-results' WHERE endpoint_id='ddw2lke';",
        ),
        (
            "ep123",
            [("app1", "res1")],
            "path/to/app-results",
            ["result_value", "result_status", "result_kind"],
            (
                "SELECT result_value,result_status,result_kind "
                "FROM 'path/to/app-results' WHERE endpoint_id='ep123' "
                "AND ((application_name='app1' AND result_name='res1'));"
            ),
        ),
        (
            "ep123",
            [("app1", "res1"), ("app1", "res2"), ("app2", "res1")],
            "app-results",
            ["result_value", "result_status", "result_kind"],
            (
                "SELECT result_value,result_status,result_kind "
                "FROM 'app-results' WHERE endpoint_id='ep123' AND "
                "((application_name='app1' AND result_name='res1') OR "
                "(application_name='app1' AND result_name='res2') OR "
                "(application_name='app2' AND result_name='res1'));"
            ),
        ),
    ],
)
def test_tsdb_query(
    endpoint_id: str,
    names: list[tuple[str, str]],
    table_path: str,
    expected_query: str,
    columns: Optional[list[str]],
) -> None:
    assert (
        V3IOTSDBConnector._get_sql_query(
            endpoint_id=endpoint_id,
            metric_and_app_names=names,
            table_path=table_path,
            columns=columns,
        )
        == expected_query
    )


def test_tsdb_predictions_existence_query() -> None:
    assert V3IOTSDBConnector._get_sql_query(
        columns=["count(latency)"],
        table_path="pipelines/metrics-data-v2/model-endpoints/predictions/",
        endpoint_id="d4b50a7727d65c7f73c33590f6fe87a40d93af2a",
    ) == (
        "SELECT count(latency) FROM 'pipelines/metrics-data-v2/model-endpoints/predictions/' "
        "WHERE endpoint_id='d4b50a7727d65c7f73c33590f6fe87a40d93af2a';"
    )


@pytest.fixture
def tsdb_df() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            (
                pd.Timestamp("2024-04-02 18:00:28", tz="UTC"),
                "histogram-data-drift",
                "70450e1ef7cc9506d42369aeeb056eaaaa0bb8bd",
                0,
                "kld_mean",
                -1.0,
                0.06563064,
                "2024-04-02 17:59:28.000000+00:00",
            ),
            (
                pd.Timestamp("2024-04-02 18:00:28", tz="UTC"),
                "histogram-data-drift",
                "70450e1ef7cc9506d42369aeeb056eaaaa0bb8bd",
                0,
                "general_drift",
                0.0,
                0.04651495,
                "2024-04-02 17:59:28.000000+00:00",
            ),
        ],
        index="time",
        columns=[
            "time",
            "application_name",
            "endpoint_id",
            "result_kind",
            "result_name",
            "result_status",
            "result_value",
            "start_infer_time",
        ],
    )


@pytest.fixture
def predictions_df() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            (
                pd.Timestamp("2024-04-02 18:00:00", tz="UTC"),
                5,
            ),
            (pd.Timestamp("2024-04-02 18:01:00", tz="UTC"), 10),
        ],
        index="time",
        columns=[
            "time",
            "count(latency)",
        ],
    )


@pytest.fixture
def _mock_frames_client(tsdb_df: pd.DataFrame) -> Iterator[None]:
    frames_client_mock = Mock()
    frames_client_mock.read = Mock(return_value=tsdb_df)

    with patch.object(
        mlrun.utils.v3io_clients, "get_frames_client", return_value=frames_client_mock
    ):
        yield


@pytest.fixture
def _mock_frames_client_predictions(predictions_df: pd.DataFrame) -> Iterator[None]:
    frames_client_mock = Mock()
    frames_client_mock.read = Mock(return_value=predictions_df)

    with patch.object(
        mlrun.utils.v3io_clients, "get_frames_client", return_value=frames_client_mock
    ):
        yield


@pytest.mark.usefixtures("_mock_frames_client")
def test_read_results_data() -> None:
    tsdb_connector = V3IOTSDBConnector(project="fictitious-one")
    data = tsdb_connector.read_metrics_data(
        endpoint_id="70450e1ef7cc9506d42369aeeb056eaaaa0bb8bd",
        start=datetime(2024, 4, 2, 18, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 4, 3, 18, 0, 0, tzinfo=timezone.utc),
        metrics=[
            ModelEndpointMonitoringMetric(
                project="fictitious-one",
                app="histogram-data-drift",
                name="kld_mean",
                full_name="fictitious-one.histogram-data-drift.result.kld_mean",
                type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
            ),
            ModelEndpointMonitoringMetric(
                project="fictitious-one",
                app="histogram-data-drift",
                name="general_drift",
                full_name="fictitious-one.histogram-data-drift.result.general_drift",
                type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
            ),
            ModelEndpointMonitoringMetric(
                project="fictitious-one",
                app="late-app",
                name="notfound",
                full_name="fictitious-one.late-app.result.notfound",
                type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
            ),
        ],
    )
    assert len(data) == 3
    counter = Counter([type(values) for values in data])
    assert counter[ModelEndpointMonitoringResultValues] == 2
    assert counter[ModelEndpointMonitoringMetricNoData] == 1


@pytest.mark.usefixtures("_mock_frames_client_predictions")
def test_read_predictions() -> None:
    predictions_args = {
        "endpoint_id": "70450e1ef7cc9506d42369aeeb056eaaaa0bb8bd",
        "start": datetime(2024, 4, 2, 18, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2024, 4, 3, 18, 0, 0, tzinfo=timezone.utc),
        "aggregation_window": "1m",
    }

    tsdb_connector = V3IOTSDBConnector(project="fictitious-one")
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as err:
        tsdb_connector.read_predictions(**predictions_args)
        assert (
            str(err.value)
            == "both or neither of `aggregation_window` and `agg_funcs` must be provided"
        )
    predictions_args["agg_funcs"] = ["count"]
    result = tsdb_connector.read_predictions(**predictions_args)
    assert result.full_name == "fictitious-one.mlrun-infra.metric.invocations"
    assert result.values == [
        _MetricPoint(
            timestamp=pd.Timestamp("2024-04-02 18:00:00", tz="UTC"),
            value=5.0,
        ),
        _MetricPoint(
            timestamp=pd.Timestamp("2024-04-02 18:01:00", tz="UTC"),
            value=10.0,
        ),
    ]


@pytest.mark.parametrize(
    ("input_event", "expected_output"),
    [
        ({}, {}),
        (
            {"": 1, "1": 2, "f3": 3, "my--pred": 0.2},
            {"": 1, "_1": 2, "f3": 3, "my__pred": 0.2},
        ),
    ],
)
def test_normalize_dict_for_v3io_frames(
    input_event: dict[str, Any], expected_output: dict[str, Any]
) -> None:
    assert _normalize_dict_for_v3io_frames(input_event) == expected_output
