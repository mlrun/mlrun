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
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import v3io_frames

import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.utils.v3io_clients
from mlrun.common.schemas.model_monitoring.model_endpoints import (
    ModelEndpointMonitoringMetric,
    ModelEndpointMonitoringMetricNoData,
    ModelEndpointMonitoringResultValues,
    _MetricPoint,
)
from mlrun.model_monitoring.db.tsdb.v3io.stream_graph_steps import (
    _normalize_dict_for_v3io_frames,
)
from mlrun.model_monitoring.db.tsdb.v3io.v3io_connector import (
    V3IOTSDBConnector,
    _is_no_schema_error,
)


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
def metric_event() -> dict[str, Any]:
    return {
        mm_constants.WriterEvent.ENDPOINT_ID: "ep-id",
        mm_constants.WriterEvent.APPLICATION_NAME: "some-app",
        mm_constants.MetricData.METRIC_NAME: "metric_1",
        mm_constants.MetricData.METRIC_VALUE: 0.345,
        mm_constants.WriterEvent.START_INFER_TIME: "2024-05-10T13:00:00.0+00:00",
        mm_constants.WriterEvent.END_INFER_TIME: "2024-05-10T14:00:00.0+00:00",
    }


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
                "",
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
                "{'extra_data': 'some data'}",
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
            "result_extra_data",
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


@pytest.mark.parametrize(("with_result_extra_data"), [False, True])
@pytest.mark.usefixtures("_mock_frames_client")
def test_read_results_data(with_result_extra_data: bool) -> None:
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
                type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
            ),
            ModelEndpointMonitoringMetric(
                project="fictitious-one",
                app="histogram-data-drift",
                name="general_drift",
                type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
            ),
            ModelEndpointMonitoringMetric(
                project="fictitious-one",
                app="late-app",
                name="notfound",
                type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
            ),
        ],
        with_result_extra_data=with_result_extra_data,
    )
    assert len(data) == 3
    counter = Counter([type(values) for values in data])
    assert counter[ModelEndpointMonitoringResultValues] == 2
    assert counter[ModelEndpointMonitoringMetricNoData] == 1
    if with_result_extra_data:
        assert data[0].values[0].extra_data == "{'extra_data': 'some data'}"
    else:
        assert data[0].values[0].extra_data == ""


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
