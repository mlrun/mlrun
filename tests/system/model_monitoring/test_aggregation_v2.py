# Copyright 2025 Iguazio
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

"""
System tests for v2 metrics-values API with pre-aggregation support (ML-11445).

These tests verify that the v2 API correctly aggregates time-series data
across multiple time buckets using TimescaleDB continuous aggregates.

Unlike the existing system tests that generate data through inference,
these tests insert time-distributed data directly into TSDB to properly
test aggregation across hours/days.
"""

import json
import typing
import uuid
from datetime import UTC, datetime, timedelta

import pytest

import mlrun
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.common.types
from mlrun.datastore.datastore_profile import DatastoreProfilePostgreSQL
from mlrun.model_monitoring.db.tsdb.preaggregate import PreAggregateConfig
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    TimescaleDBConnection,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_operations import (
    TimescaleDBOperationsManager,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_query_builder import (
    TimescaleDBNaming,
)
from tests.system.base import TestMLRunSystem

from . import TestMLRunSystemModelMonitoring

# Expected aggregation results for METRICS_TEST_DATA
EXPECTED_HOUR_0_AVG = 20.0  # (10 + 20 + 30) / 3
EXPECTED_HOUR_0_MIN = 10.0
EXPECTED_HOUR_0_MAX = 30.0
EXPECTED_HOUR_0_COUNT = 3

EXPECTED_HOUR_1_AVG = 150.0  # (100 + 200) / 2
EXPECTED_HOUR_1_MIN = 100.0
EXPECTED_HOUR_1_MAX = 200.0
EXPECTED_HOUR_1_COUNT = 2

EXPECTED_HOUR_2_AVG = 50.0
EXPECTED_HOUR_2_MIN = 50.0
EXPECTED_HOUR_2_MAX = 50.0
EXPECTED_HOUR_2_COUNT = 1

# Expected status for RESULTS_TEST_DATA (MAX = worst = DETECTED)
EXPECTED_MAX_STATUS = 2  # DETECTED


@TestMLRunSystemModelMonitoring.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestAggregationV2API(TestMLRunSystemModelMonitoring):
    """System tests for v2 metrics-values API with pre-aggregation."""

    project_name = "test-aggregation-v2"

    # Test data: {hour_offset: [(minute_offset, value), ...]}
    METRICS_TEST_DATA: typing.ClassVar[dict[int, list[tuple[int, float]]]] = {
        0: [(10, 10.0), (30, 20.0), (50, 30.0)],
        1: [(15, 100.0), (45, 200.0)],
        2: [(30, 50.0)],
    }

    # Results test data with status escalation
    # Status values: NO_DETECTION=0, POSSIBLE_DETECTION=1, DETECTED=2
    RESULTS_TEST_DATA: typing.ClassVar[dict[int, list[tuple[int, float, int, int]]]] = {
        0: [
            (10, 0.1, 0, 0),  # NO_DETECTION
            (30, 0.5, 1, 0),  # POSSIBLE_DETECTION
            (50, 0.9, 2, 0),  # DETECTED
        ],
    }

    @classmethod
    def custom_setup_class(cls) -> None:
        cls.run_db = mlrun.get_run_db()
        cls.endpoint_id = f"test-agg-ep-{uuid.uuid4().hex[:8]}"
        cls.app_name = "test-aggregation-app"
        cls.metric_name = "latency"
        cls.result_name = "drift_score"

    def custom_setup(self) -> None:
        self.set_mm_credentials()
        super(TestMLRunSystem, self).custom_setup(project_name=self.project_name)
        self._setup_tsdb_with_time_distributed_data()

    def _get_tsdb_connection(self) -> TimescaleDBConnection:
        """Get a TimescaleDB connection from the profile."""
        if not isinstance(self.mm_tsdb_profile, DatastoreProfilePostgreSQL):
            pytest.skip("Test requires PostgreSQL/TimescaleDB TSDB profile")

        dsn = self.mm_tsdb_profile.get_connection_string()
        return TimescaleDBConnection(dsn, max_connections=1, autocommit=False)

    def _get_autocommit_connection(self) -> TimescaleDBConnection:
        """Get a TimescaleDB connection with autocommit for DDL operations."""
        if not isinstance(self.mm_tsdb_profile, DatastoreProfilePostgreSQL):
            pytest.skip("Test requires PostgreSQL/TimescaleDB TSDB profile")

        dsn = self.mm_tsdb_profile.get_connection_string()
        return TimescaleDBConnection(dsn, max_connections=1, autocommit=True)

    def _setup_tsdb_with_time_distributed_data(self) -> None:
        """Insert time-distributed metrics/results directly into TSDB."""
        connection = self._get_tsdb_connection()

        pre_aggregate_config = PreAggregateConfig(
            aggregate_intervals=["1h"],
            agg_functions=["avg", "min", "max", "count"],
            retention_policy={
                "raw": "7 days",
                "1h": "30 days",
            },
        )

        self._operations_handler = TimescaleDBOperationsManager(
            project=self.project_name,
            connection=connection,
            pre_aggregate_config=pre_aggregate_config,
        )
        self._operations_handler.create_tables()

        now = datetime.now(UTC)
        self._base_time = now - timedelta(hours=5)

        self._insert_metrics_data()
        self._insert_results_data()
        self._refresh_continuous_aggregates()
        self._connection = connection

    def _insert_metrics_data(self) -> None:
        """Insert metrics test data into TSDB."""
        for hour_offset, values in self.METRICS_TEST_DATA.items():
            for minute_offset, value in values:
                timestamp = self._base_time + timedelta(
                    hours=hour_offset, minutes=minute_offset
                )
                event = {
                    mm_schemas.WriterEvent.END_INFER_TIME: timestamp,
                    mm_schemas.WriterEvent.START_INFER_TIME: timestamp,
                    mm_schemas.WriterEvent.ENDPOINT_ID: self.endpoint_id,
                    mm_schemas.WriterEvent.APPLICATION_NAME: self.app_name,
                    mm_schemas.MetricData.METRIC_NAME: self.metric_name,
                    mm_schemas.MetricData.METRIC_VALUE: value,
                }
                self._operations_handler.write_application_event(
                    event, kind=mm_schemas.WriterEventKind.METRIC
                )

    def _insert_results_data(self) -> None:
        """Insert results test data into TSDB."""
        for hour_offset, values in self.RESULTS_TEST_DATA.items():
            for minute_offset, value, status, kind in values:
                timestamp = self._base_time + timedelta(
                    hours=hour_offset, minutes=minute_offset
                )
                event = {
                    mm_schemas.WriterEvent.END_INFER_TIME: timestamp,
                    mm_schemas.WriterEvent.START_INFER_TIME: timestamp,
                    mm_schemas.WriterEvent.ENDPOINT_ID: self.endpoint_id,
                    mm_schemas.WriterEvent.APPLICATION_NAME: self.app_name,
                    mm_schemas.ResultData.RESULT_NAME: self.result_name,
                    mm_schemas.ResultData.RESULT_VALUE: value,
                    mm_schemas.ResultData.RESULT_STATUS: status,
                    mm_schemas.ResultData.RESULT_KIND: kind,
                    mm_schemas.ResultData.RESULT_EXTRA_DATA: "{}",
                }
                self._operations_handler.write_application_event(
                    event, kind=mm_schemas.WriterEventKind.RESULT
                )

    def _refresh_continuous_aggregates(self) -> None:
        """Manually refresh CAGGs for test data to be queryable."""
        autocommit_conn = self._get_autocommit_connection()
        tables = self._operations_handler.tables

        metrics_table = tables[mm_schemas.TimescaleDBTables.METRICS]
        metrics_cagg_name = TimescaleDBNaming.get_cagg_view_name(
            metrics_table.full_name(), "1h"
        )
        autocommit_conn.run(
            statements=f"CALL refresh_continuous_aggregate('{metrics_cagg_name}', NULL, NULL);"
        )

        results_table = tables[mm_schemas.TimescaleDBTables.APP_RESULTS]
        results_cagg_name = TimescaleDBNaming.get_cagg_view_name(
            results_table.full_name(), "1h"
        )
        autocommit_conn.run(
            statements=f"CALL refresh_continuous_aggregate('{results_cagg_name}', NULL, NULL);"
        )

    def _build_metric_full_name(self) -> str:
        """Build the full metric name in project.app.metric format."""
        return f"{self.project_name}.{self.app_name}.{self.metric_name}"

    def _build_result_full_name(self) -> str:
        """Build the full result name in project.app.result format."""
        return f"{self.project_name}.{self.app_name}.{self.result_name}"

    def _get_time_range_ms(self) -> tuple[int, int]:
        """Get start/end timestamps in milliseconds covering test data."""
        start = int((self._base_time - timedelta(hours=1)).timestamp() * 1000)
        end = int((self._base_time + timedelta(hours=5)).timestamp() * 1000)
        return start, end

    def custom_teardown(self) -> None:
        """Clean up TSDB resources after test."""
        if hasattr(self, "_operations_handler"):
            try:
                self._operations_handler.delete_tsdb_resources()
            except Exception:
                pass

    def test_v2_api_aggregation_1h_correct_values(self) -> None:
        """Verify 1-hour aggregation returns correct min/max/avg/count values."""
        start, end = self._get_time_range_ms()
        metric_name = self._build_metric_full_name()

        response = self.run_db.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=(
                f"v2/projects/{self.project_name}/model-endpoints/{self.endpoint_id}"
                f"/metrics-values?name={metric_name}&start={start}&end={end}"
                f"&agg-period=1h&agg-function=avg&agg-function=min&agg-function=max"
            ),
        )
        results = json.loads(response.content.decode())

        assert len(results) == 1
        result = results[0]

        assert result["data"] is True
        agg_config = result["aggregation_config"]
        assert agg_config["aggregated"] is True
        assert agg_config["period"] == "1h"
        assert set(agg_config["functions"]) == {"avg", "min", "max"}

        values = result["values"]
        expected_bucket_count = len(self.METRICS_TEST_DATA)
        assert len(values) == expected_bucket_count

        for value in values:
            assert len(value) >= 4

    def test_v2_api_raw_returns_all_points(self) -> None:
        """Verify agg-period=raw returns individual data points."""
        start, end = self._get_time_range_ms()
        metric_name = self._build_metric_full_name()

        response = self.run_db.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=(
                f"v2/projects/{self.project_name}/model-endpoints/{self.endpoint_id}"
                f"/metrics-values?name={metric_name}&start={start}&end={end}"
                f"&agg-period=raw"
            ),
        )
        results = json.loads(response.content.decode())

        assert len(results) == 1
        result = results[0]

        agg_config = result["aggregation_config"]
        assert agg_config["aggregated"] is False

        values = result["values"]
        expected_count = sum(len(v) for v in self.METRICS_TEST_DATA.values())
        assert len(values) == expected_count

        for value in values:
            assert len(value) == 2

    def test_v2_api_auto_select_period(self) -> None:
        """Verify auto-selection when no agg-period specified."""
        start, end = self._get_time_range_ms()
        metric_name = self._build_metric_full_name()

        response = self.run_db.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=(
                f"v2/projects/{self.project_name}/model-endpoints/{self.endpoint_id}"
                f"/metrics-values?name={metric_name}&start={start}&end={end}"
            ),
        )
        results = json.loads(response.content.decode())

        assert len(results) == 1
        result = results[0]

        agg_config = result["aggregation_config"]
        assert "aggregated" in agg_config

        if agg_config["aggregated"]:
            assert agg_config["period"] is not None

    def test_v2_api_results_with_status_aggregation(self) -> None:
        """Verify result_status uses MAX (worst status in window)."""
        start, end = self._get_time_range_ms()
        result_name = self._build_result_full_name()

        response = self.run_db.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=(
                f"v2/projects/{self.project_name}/model-endpoints/{self.endpoint_id}"
                f"/metrics-values?name={result_name}&start={start}&end={end}"
                f"&agg-period=1h&agg-function=avg&agg-function=max"
            ),
        )
        results = json.loads(response.content.decode())

        result = next(
            (r for r in results if r["full_name"] == result_name),
            None,
        )
        assert result is not None

        if result["data"]:
            agg_config = result["aggregation_config"]
            assert agg_config["aggregated"] is True
            values = result["values"]
            assert len(values) >= 1

    def test_v2_api_multiple_metrics_same_request(self) -> None:
        """Verify multiple metrics in single request."""
        start, end = self._get_time_range_ms()
        metric_name = self._build_metric_full_name()
        result_name = self._build_result_full_name()

        response = self.run_db.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=(
                f"v2/projects/{self.project_name}/model-endpoints/{self.endpoint_id}"
                f"/metrics-values?name={metric_name}&name={result_name}"
                f"&start={start}&end={end}&agg-period=1h&agg-function=avg"
            ),
        )
        results = json.loads(response.content.decode())

        assert len(results) == 2

        full_names = {r["full_name"] for r in results}
        assert metric_name in full_names
        assert result_name in full_names

    def test_v2_api_response_format_matches_schema(self) -> None:
        """Verify response matches v2 schema structure."""
        start, end = self._get_time_range_ms()
        metric_name = self._build_metric_full_name()

        response = self.run_db.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=(
                f"v2/projects/{self.project_name}/model-endpoints/{self.endpoint_id}"
                f"/metrics-values?name={metric_name}&start={start}&end={end}"
                f"&agg-period=1h&agg-function=avg"
            ),
        )
        results = json.loads(response.content.decode())

        assert len(results) == 1
        result = results[0]

        assert result["full_name"] == metric_name
        assert "type" in result
        assert "data" in result
        assert "aggregation_config" in result

        agg_config = result["aggregation_config"]
        assert agg_config["aggregated"] is True
        assert agg_config["period"] == "1h"
        assert "avg" in agg_config["functions"]

        if result["data"]:
            values = result["values"]
            assert isinstance(values, list)
            if values:
                assert isinstance(values[0], list)

    def test_v2_api_empty_time_range_returns_no_data(self) -> None:
        """Verify behavior when no data in requested time range."""
        future = datetime.now(UTC) + timedelta(days=365)
        start = int(future.timestamp() * 1000)
        end = int((future + timedelta(days=1)).timestamp() * 1000)
        metric_name = self._build_metric_full_name()

        response = self.run_db.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=(
                f"v2/projects/{self.project_name}/model-endpoints/{self.endpoint_id}"
                f"/metrics-values?name={metric_name}&start={start}&end={end}"
                f"&agg-period=1h&agg-function=avg"
            ),
        )
        results = json.loads(response.content.decode())

        assert len(results) == 1
        result = results[0]

        assert result["data"] is False
