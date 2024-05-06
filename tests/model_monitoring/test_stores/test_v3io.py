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

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Optional, TypedDict
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import v3io.dataplane.output
import v3io.dataplane.response

import mlrun.utils.v3io_clients
from mlrun.common.schemas.model_monitoring import (
    ModelEndpointMonitoringMetric,
    ModelEndpointMonitoringMetricType,
)
from mlrun.model_monitoring.db.stores.v3io_kv.kv_store import KVStoreBase
from mlrun.model_monitoring.db.v3io_tsdb_reader import _get_sql_query, read_data


@pytest.fixture
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
        ("default", [{"__name": ".#schema"}], []),
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
                    type=ModelEndpointMonitoringMetricType.RESULT,
                    name="some-metric-name1",
                    full_name="default.some-app-name.result.some-metric-name1",
                ),
                ModelEndpointMonitoringMetric(
                    project="default",
                    app="some-app-name",
                    type=ModelEndpointMonitoringMetricType.RESULT,
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
                },
                {"__name": ".#schema"},
            ],
            [
                ModelEndpointMonitoringMetric(
                    project="project-1",
                    app="app1",
                    type=ModelEndpointMonitoringMetricType.RESULT,
                    name="drift-res",
                    full_name="project-1.app1.result.drift-res",
                )
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
            {"__name": ".#schema"},
        ],
    )
    METRICS = [
        ModelEndpointMonitoringMetric(
            project=PROJECT,
            app="perf-app",
            type=ModelEndpointMonitoringMetricType.RESULT,
            name="latency-sec",
            full_name="demo-proj.perf-app.result.latency-sec",
        ),
        ModelEndpointMonitoringMetric(
            project=PROJECT,
            app="model-as-a-judge-app",
            type=ModelEndpointMonitoringMetricType.RESULT,
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
        assert store.get_model_endpoint_metrics(endpoint_id) == expected_metrics

    @classmethod
    def test_response_error(cls, store_with_err: KVStoreBase) -> None:
        """Test that non 404 errors are not silenced"""
        with pytest.raises(v3io.dataplane.response.HttpResponseError):
            store_with_err.get_model_endpoint_metrics(cls.ENDPOINT)


@pytest.mark.parametrize(
    ("endpoint_id", "names", "expected_query"),
    [
        ("ddw2lke", [], "SELECT * FROM 'app-results' WHERE endpoint_id='ddw2lke';"),
        (
            "ep123",
            [("app1", "res1")],
            (
                "SELECT * FROM 'app-results' WHERE endpoint_id='ep123' "
                "AND ((application_name='app1' AND result_name='res1'));"
            ),
        ),
        (
            "ep123",
            [("app1", "res1"), ("app1", "res2"), ("app2", "res1")],
            (
                "SELECT * FROM 'app-results' WHERE endpoint_id='ep123' AND "
                "((application_name='app1' AND result_name='res1') OR "
                "(application_name='app1' AND result_name='res2') OR "
                "(application_name='app2' AND result_name='res1'));"
            ),
        ),
    ],
)
def test_tsdb_query(
    endpoint_id: str, names: list[tuple[str, str]], expected_query: str
) -> None:
    assert _get_sql_query(endpoint_id, names) == expected_query


@pytest.fixture
def tsdb_df() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            (
                pd.Timestamp("2024-04-02 18:00:28+0000", tz="UTC"),
                "histogram-data-drift",
                "70450e1ef7cc9506d42369aeeb056eaaaa0bb8bd",
                "ResultKindApp.data_drift",
                "kld_mean",
                -1.0,
                0.06563064,
                "2024-04-02 17:59:28.000000+00:00",
            ),
            (
                pd.Timestamp("2024-04-02 18:00:28+0000", tz="UTC"),
                "histogram-data-drift",
                "70450e1ef7cc9506d42369aeeb056eaaaa0bb8bd",
                "ResultKindApp.data_drift",
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
def _mock_frames_client(tsdb_df: pd.DataFrame) -> Iterator[None]:
    frames_client_mock = Mock()
    frames_client_mock.read = Mock(return_value=tsdb_df)

    with patch.object(
        mlrun.utils.v3io_clients, "get_frames_client", return_value=frames_client_mock
    ):
        yield


@pytest.mark.usefixtures("_mock_frames_client")
def test_read_data() -> None:
    data = read_data(
        project="fictitious-one",
        endpoint_id="70450e1ef7cc9506d42369aeeb056eaaaa0bb8bd",
        start=datetime(2024, 4, 2, 18, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 4, 3, 18, 0, 0, tzinfo=timezone.utc),
        names=[
            ("histogram-data-drift", "kld_mean"),
            ("histogram-data-drift", "general_drift"),
        ],
    )
    assert len(data) == 2
