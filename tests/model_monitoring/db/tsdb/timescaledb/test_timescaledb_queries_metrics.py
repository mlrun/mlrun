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

import os
from datetime import datetime

import pandas as pd
import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas

# Skip entire module if connection string is not available or not PostgreSQL
connection_string = os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__TSDB_CONNECTION")
pytestmark = pytest.mark.skipif(
    not connection_string or not connection_string.startswith("postgres"),
    reason="TimescaleDB connection string not available or not PostgreSQL",
)


class TestMetricsQueries:
    """Tests for metrics-related query operations."""

    def test_get_model_endpoint_real_time_metrics_empty(self, query_handler):
        """Test get_model_endpoint_real_time_metrics with no data."""
        result = query_handler.get_model_endpoint_real_time_metrics(
            endpoint_id="nonexistent_endpoint",
            metrics=["accuracy", "precision"],
            start="2024-01-01T00:00:00",
            end="2024-01-02T00:00:00",
        )

        assert isinstance(result, dict)
        # When there's no data, the result should be an empty dict
        assert len(result) == 0

    def test_get_model_endpoint_real_time_metrics_with_data(self, query_handler):
        """Test get_model_endpoint_real_time_metrics with sample data."""
        # Write some test metrics first
        test_metrics = [
            {
                mm_schemas.WriterEvent.END_INFER_TIME: datetime(2024, 1, 15, 12, 0, 0),
                mm_schemas.WriterEvent.START_INFER_TIME: datetime(
                    2024, 1, 15, 12, 0, 0
                ),
                mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint",
                mm_schemas.WriterEvent.APPLICATION_NAME: "test_app",
                mm_schemas.MetricData.METRIC_NAME: "accuracy",
                mm_schemas.MetricData.METRIC_VALUE: 0.95,
            },
            {
                mm_schemas.WriterEvent.END_INFER_TIME: datetime(2024, 1, 15, 12, 5, 0),
                mm_schemas.WriterEvent.START_INFER_TIME: datetime(
                    2024, 1, 15, 12, 5, 0
                ),
                mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint",
                mm_schemas.WriterEvent.APPLICATION_NAME: "test_app",
                mm_schemas.MetricData.METRIC_NAME: "precision",
                mm_schemas.MetricData.METRIC_VALUE: 0.87,
            },
        ]

        for metric in test_metrics:
            query_handler.write_application_event(
                metric, mm_schemas.WriterEventKind.METRIC
            )

        result = query_handler.get_model_endpoint_real_time_metrics(
            endpoint_id="test_endpoint",
            metrics=["accuracy", "precision"],
            start="2024-01-15T00:00:00",
            end="2024-01-16T00:00:00",
        )

        assert isinstance(result, dict)
        # The method returns results with "default_metric" as the key
        assert "default_metric" in result
        assert isinstance(result["default_metric"], list)
        # Verify actual data points are returned
        data_points = result["default_metric"]
        assert len(data_points) == 2  # We inserted 2 metrics for test_endpoint
        # Each data point should be a tuple of (timestamp, value)
        for timestamp_str, value in data_points:
            assert isinstance(timestamp_str, str)
            assert isinstance(value, (int, float))
            assert value in [0.95, 0.87]  # Should match our test values

    def test_read_metrics_data_method_exists(self, query_handler):
        """Test that read_metrics_data method exists and is callable."""
        # The method requires complex ModelEndpointMonitoringMetric objects
        # which are difficult to construct in tests
        assert hasattr(query_handler, "read_metrics_data")
        assert callable(getattr(query_handler, "read_metrics_data"))

    def test_get_metrics_metadata(self, query_handler):
        """Test get_metrics_metadata method."""
        # First insert some metrics data to ensure we have metadata
        test_metrics = [
            {
                mm_schemas.WriterEvent.END_INFER_TIME: datetime(2024, 1, 15, 12, 0, 0),
                mm_schemas.WriterEvent.START_INFER_TIME: datetime(
                    2024, 1, 15, 12, 0, 0
                ),
                mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint",
                mm_schemas.WriterEvent.APPLICATION_NAME: "test_app",
                mm_schemas.MetricData.METRIC_NAME: "accuracy",
                mm_schemas.MetricData.METRIC_VALUE: 0.95,
            },
            {
                mm_schemas.WriterEvent.END_INFER_TIME: datetime(2024, 1, 15, 12, 5, 0),
                mm_schemas.WriterEvent.START_INFER_TIME: datetime(
                    2024, 1, 15, 12, 5, 0
                ),
                mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint",
                mm_schemas.WriterEvent.APPLICATION_NAME: "test_app",
                mm_schemas.MetricData.METRIC_NAME: "precision",
                mm_schemas.MetricData.METRIC_VALUE: 0.87,
            },
        ]

        for metric in test_metrics:
            query_handler.write_application_event(
                metric, mm_schemas.WriterEventKind.METRIC
            )

        result = query_handler.get_metrics_metadata(endpoint_id="test_endpoint")

        assert isinstance(result, pd.DataFrame)

        # Should have metric_name column and verify our test metrics appear
        assert "metric_name" in result.columns
        metric_names = result["metric_name"].unique()
        assert (
            len(metric_names) == 2
        )  # We inserted 2 unique metrics: accuracy and precision

        # Should have endpoint_id column and verify it matches our query
        assert "endpoint_id" in result.columns
        endpoints = result["endpoint_id"].unique()
        assert "test_endpoint" in endpoints


class TestMetadataMethods:
    """Test metadata retrieval methods."""

    def test_get_metrics_metadata_with_data(self, query_handler):
        """Test get_metrics_metadata returns correct metadata."""
        connection = query_handler._connection
        metrics_table = query_handler.tables[mm_schemas.TimescaleDBTables.METRICS]

        # Insert test metrics data with different metric names
        test_time = datetime(2024, 1, 15, 12, 0, 0)
        metrics_data = [
            ("endpoint_1", test_time, "app1", "accuracy", 0.95),
            ("endpoint_1", test_time, "app1", "precision", 0.87),
            ("endpoint_2", test_time, "app2", "recall", 0.92),
            ("endpoint_2", test_time, "app2", "f1_score", 0.89),
        ]

        for (
            endpoint_id,
            metric_time,
            app_name,
            metric_name,
            metric_value,
        ) in metrics_data:
            connection.run(
                statements=[
                    f"""
                    INSERT INTO {metrics_table.full_name()}
                    (end_infer_time, start_infer_time, endpoint_id, application_name, metric_name, metric_value)
                    VALUES ('{metric_time}', '{metric_time}', '{endpoint_id}',
                            '{app_name}', '{metric_name}', {metric_value})
                    """
                ]
            )

        result = query_handler.get_metrics_metadata(
            endpoint_id=["endpoint_1", "endpoint_2"],
            start=datetime(2024, 1, 15, 11, 0, 0),
            end=datetime(2024, 1, 15, 13, 0, 0),
        )

        assert isinstance(result, pd.DataFrame)
        assert "metric_name" in result.columns

        # Verify we have all the metric names we inserted
        metric_names = set(result["metric_name"].unique())
        expected_names = {"accuracy", "precision", "recall", "f1_score"}
        assert metric_names == expected_names

    def test_get_metrics_metadata_empty(self, query_handler):
        """Test get_metrics_metadata returns empty DataFrame when no data exists."""
        result = query_handler.get_metrics_metadata(
            endpoint_id=["nonexistent_endpoint"],
            start=datetime(2024, 1, 15, 11, 0, 0),
            end=datetime(2024, 1, 15, 13, 0, 0),
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    # COMMENTED OUT: Redundant results metadata tests - belong logically in test_timescaledb_queries_results.py
    # def test_get_results_metadata_with_data(self, query_handler):
    #     """Test get_results_metadata returns correct metadata."""
    #     connection = query_handler._connection
    #     results_table = query_handler.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]

    #     # Insert test results data with different result names
    #     test_time = datetime(2024, 1, 15, 12, 0, 0)
    #     results_data = [
    #         ("endpoint_1", test_time, "app1", "drift_detection", 0.85, 1, 1),
    #         ("endpoint_1", test_time, "app1", "data_quality", 0.92, 1, 2),
    #         ("endpoint_2", test_time, "app2", "model_performance", 0.88, 1, 3),
    #         ("endpoint_2", test_time, "app2", "anomaly_detection", 0.76, 2, 4),
    #     ]

    #     for endpoint_id, result_time, app_name, result_name, result_value, result_status, result_kind in results_data:
    #         connection.run(
    #             statements=[
    #                 f"""
    #                 INSERT INTO {results_table.full_name()}
    #                 (end_infer_time, start_infer_time, endpoint_id, application_name,
    #                  result_name, result_value, result_status, result_kind)
    #                 VALUES ('{result_time}', '{result_time}', '{endpoint_id}',
    #                         '{app_name}', '{result_name}', {result_value}, {result_status}, {result_kind})
    #                 """
    #             ]
    #         )

    #     result = query_handler.get_results_metadata(
    #         endpoint_id=["endpoint_1", "endpoint_2"],
    #         start=datetime(2024, 1, 15, 11, 0, 0),
    #         end=datetime(2024, 1, 15, 13, 0, 0),
    #     )

    #     assert isinstance(result, pd.DataFrame)
    #     assert "result_name" in result.columns

    #     # Verify we have all the result names we inserted
    #     result_names = set(result["result_name"].unique())
    #     expected_names = {"drift_detection", "data_quality", "model_performance", "anomaly_detection"}
    #     assert result_names == expected_names

    # def test_get_results_metadata_empty(self, query_handler):
    #     """Test get_results_metadata returns empty DataFrame when no data exists."""
    #     result = query_handler.get_results_metadata(
    #         endpoint_id=["nonexistent_endpoint"],
    #         start=datetime(2024, 1, 15, 11, 0, 0),
    #         end=datetime(2024, 1, 15, 13, 0, 0),
    #     )

    #     assert isinstance(result, pd.DataFrame)
    #     assert result.empty
