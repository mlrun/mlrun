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

from datetime import datetime

import pandas as pd

import mlrun.common.schemas.model_monitoring as mm_schemas


class TestMetricsQueries:
    """Tests for TimescaleDBMetricsQueries class - direct testing approach."""

    def test_get_model_endpoint_real_time_metrics_empty(self, query_test_helper):
        """Test get_model_endpoint_real_time_metrics with no data using direct class instantiation."""
        # Create metrics query handler using test helper
        metrics_handler = query_test_helper.create_metrics_handler()

        # Test with no data
        result = metrics_handler.get_model_endpoint_real_time_metrics(
            endpoint_id="nonexistent_endpoint",
            metrics=["accuracy", "precision"],
            start="2024-01-01T00:00:00",
            end="2024-01-02T00:00:00",
        )

        # When there's no data, the result should contain empty lists for each requested metric
        assert len(result) == 2  # Two metrics requested
        assert result["accuracy"] == []  # No data for accuracy
        assert result["precision"] == []  # No data for precision

    def test_get_model_endpoint_real_time_metrics_with_data(self, query_test_helper):
        """Test get_model_endpoint_real_time_metrics with sample data."""
        # Create metrics handler using test helper
        metrics_handler = query_test_helper.create_metrics_handler()

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
            query_test_helper.write_application_event(
                metric, mm_schemas.WriterEventKind.METRIC
            )

        result = metrics_handler.get_model_endpoint_real_time_metrics(
            endpoint_id="test_endpoint",
            metrics=["accuracy", "precision"],
            start="2024-01-15T00:00:00",
            end="2024-01-16T00:00:00",
        )

        assert isinstance(result, dict)

        # Verify accuracy data
        accuracy_data = result["accuracy"]
        assert len(accuracy_data) == 1  # One accuracy metric inserted
        timestamp_str, value = accuracy_data[0]
        assert "2024-01-15T12:00:00" in timestamp_str  # Should match inserted timestamp
        assert value == 0.95  # Should match our test value

        # Verify precision data
        precision_data = result["precision"]
        timestamp_str, value = precision_data[0]
        assert "2024-01-15T12:05:00" in timestamp_str  # Should match inserted timestamp
        assert value == 0.87  # Should match our test value

    def test_get_metrics_metadata(self, query_test_helper):
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

        # Create metrics handler using test helper
        metrics_handler = query_test_helper.create_metrics_handler()

        for metric in test_metrics:
            query_test_helper.write_application_event(
                metric, mm_schemas.WriterEventKind.METRIC
            )

        result = metrics_handler.get_metrics_metadata(endpoint_id="test_endpoint")

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

    def test_get_metrics_metadata_with_data(self, query_test_helper):
        """Test get_metrics_metadata returns correct metadata."""
        # Create metrics handler using test helper
        metrics_handler = query_test_helper.create_metrics_handler()

        metrics_table = query_test_helper.table_schemas[
            mm_schemas.TimescaleDBTables.METRICS
        ]

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
            query_test_helper.connection.run(
                statements=[
                    f"""
                    INSERT INTO {metrics_table.full_name()}
                    (end_infer_time, start_infer_time, endpoint_id, application_name, metric_name, metric_value)
                    VALUES ('{metric_time}', '{metric_time}', '{endpoint_id}',
                            '{app_name}', '{metric_name}', {metric_value})
                    """
                ]
            )

        result = metrics_handler.get_metrics_metadata(
            endpoint_id=["endpoint_1", "endpoint_2"],
            start=datetime(2024, 1, 15, 11, 0, 0),
            end=datetime(2024, 1, 15, 13, 0, 0),
        )

        assert isinstance(result, pd.DataFrame)

        # Verify we have all the metric names we inserted
        metric_names = set(result["metric_name"].unique())
        expected_names = {"accuracy", "precision", "recall", "f1_score"}
        assert metric_names == expected_names

    def test_get_metrics_metadata_empty(self, query_test_helper):
        """Test get_metrics_metadata returns empty DataFrame when no data exists."""
        # Create metrics handler using test helper
        metrics_handler = query_test_helper.create_metrics_handler()

        result = metrics_handler.get_metrics_metadata(
            endpoint_id=["nonexistent_endpoint"],
            start=datetime(2024, 1, 15, 11, 0, 0),
            end=datetime(2024, 1, 15, 13, 0, 0),
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_read_metrics_data_fallback_mechanism(self, query_test_helper):
        """Test that read_metrics_data uses fallback when pre-aggregate query fails."""
        # Create metrics handler using test helper
        metrics_handler = query_test_helper.create_metrics_handler()

        # Create test metrics to query for
        test_metrics = [
            mm_schemas.ModelEndpointMonitoringMetric(
                full_name=f"{query_test_helper.project_name}.test-app.metric.accuracy",
                type="metric",
                project=query_test_helper.project_name,
                app="test-app",
                name="accuracy",
            )
        ]

        # Test with no data - should use fallback mechanism and return empty DataFrame
        result_df = metrics_handler.read_metrics_data_impl(
            endpoint_id="test-endpoint",
            start=datetime(2024, 1, 15, 11, 0, 0),
            end=datetime(2024, 1, 15, 13, 0, 0),
            metrics=test_metrics,
        )

        # Should return a DataFrame (even if empty)
        import pandas as pd

        assert isinstance(result_df, pd.DataFrame)

        # Should be empty DataFrame since we didn't insert any data
        assert result_df.empty

        # Now test with actual data to ensure the method works correctly
        # Insert test metrics data
        test_metric_data = {
            mm_schemas.WriterEvent.END_INFER_TIME: datetime(2024, 1, 15, 12, 0, 0),
            mm_schemas.WriterEvent.START_INFER_TIME: datetime(2024, 1, 15, 12, 0, 0),
            mm_schemas.WriterEvent.ENDPOINT_ID: "test-endpoint",
            mm_schemas.WriterEvent.APPLICATION_NAME: "test-app",
            mm_schemas.MetricData.METRIC_NAME: "accuracy",
            mm_schemas.MetricData.METRIC_VALUE: 0.95,
        }

        query_test_helper.write_application_event(
            test_metric_data, mm_schemas.WriterEventKind.METRIC
        )

        # Query again with the same parameters - should now find data
        result_df_with_data = metrics_handler.read_metrics_data_impl(
            endpoint_id="test-endpoint",
            start=datetime(2024, 1, 15, 11, 0, 0),
            end=datetime(2024, 1, 15, 13, 0, 0),
            metrics=test_metrics,
        )

        # Should return a DataFrame with data
        assert isinstance(result_df_with_data, pd.DataFrame)
        assert not result_df_with_data.empty  # Should have data now

        # Should have the expected columns
        expected_columns = ["application_name", "metric_name", "metric_value"]
        for col in expected_columns:
            assert col in result_df_with_data.columns

        # Should have at least one row of data
        assert len(result_df_with_data) > 0

        # Verify the data content
        assert result_df_with_data["application_name"].iloc[0] == "test-app"
        assert result_df_with_data["metric_name"].iloc[0] == "accuracy"
