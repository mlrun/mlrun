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

        # Verify accuracy data matches test_metrics exactly
        accuracy_data = result["accuracy"]
        assert len(accuracy_data) == 1
        timestamp_str, value = accuracy_data[0]
        expected_time = test_metrics[0][mm_schemas.WriterEvent.END_INFER_TIME]
        assert expected_time.strftime("%Y-%m-%dT%H:%M:%S") in timestamp_str
        assert value == test_metrics[0][mm_schemas.MetricData.METRIC_VALUE]

        # Verify precision data matches test_metrics exactly
        precision_data = result["precision"]
        assert len(precision_data) == 1
        timestamp_str, value = precision_data[0]
        expected_time = test_metrics[1][mm_schemas.WriterEvent.END_INFER_TIME]
        assert expected_time.strftime("%Y-%m-%dT%H:%M:%S") in timestamp_str
        assert value == test_metrics[1][mm_schemas.MetricData.METRIC_VALUE]

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

        # Verify exact metric names from test_metrics
        assert "metric_name" in result.columns
        metric_names = set(result["metric_name"].unique())
        expected_metric_names = {
            test_metrics[0][mm_schemas.MetricData.METRIC_NAME],
            test_metrics[1][mm_schemas.MetricData.METRIC_NAME],
        }
        assert metric_names == expected_metric_names

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

    def test_read_metrics_filters_by_application_name(self, query_test_helper):
        """Test that querying metrics filters by application_name and doesn't return other apps' data.

        This test verifies the fix for the bug where build_metrics_filter only filtered by
        metric_name, causing queries to return data from all applications with that metric name.
        """
        # Insert metrics for two different applications with the SAME metric name
        test_time = datetime(2024, 1, 15, 12, 0, 0)
        metrics_table = query_test_helper.table_schemas[
            mm_schemas.TimescaleDBTables.METRICS
        ]

        test_data = [
            # App1 with accuracy metric
            {
                "endpoint_id": "test_endpoint_1",
                "application_name": "monitoring-app1",
                "metric_name": "accuracy",
                "metric_value": 0.95,
            },
            # App2 with the SAME metric name
            {
                "endpoint_id": "test_endpoint_1",
                "application_name": "monitoring-app2",
                "metric_name": "accuracy",
                "metric_value": 0.75,
            },
            # App1 with different metric
            {
                "endpoint_id": "test_endpoint_1",
                "application_name": "monitoring-app1",
                "metric_name": "precision",
                "metric_value": 0.88,
            },
        ]

        # Insert test data
        for data in test_data:
            query_test_helper.connection.run(
                statements=[
                    f"""
                    INSERT INTO {metrics_table.full_name()}
                    (end_infer_time, start_infer_time, endpoint_id, application_name, metric_name,
                     metric_value)
                    VALUES ('{test_time}', '{test_time}', '{data["endpoint_id"]}',
                            '{data["application_name"]}', '{data["metric_name"]}',
                            {data["metric_value"]})
                    """
                ]
            )

        # Query for ONLY monitoring-app1's accuracy metric
        metrics_handler = query_test_helper.create_metrics_handler()
        test_metrics = [
            mm_schemas.ModelEndpointMonitoringMetric(
                project=query_test_helper.project_name,
                app="monitoring-app1",
                name="accuracy",
                type=mm_schemas.ModelEndpointMonitoringMetricType.METRIC,
            )
        ]

        result = metrics_handler.read_metrics_data_impl(
            endpoint_id="test_endpoint_1",
            start=datetime(2024, 1, 15, 0, 0, 0),
            end=datetime(2024, 1, 15, 23, 59, 59),
            metrics=test_metrics,
        )

        # Verify we ONLY get monitoring-app1's data, NOT monitoring-app2's data
        assert not result.empty, "Should have returned data for monitoring-app1"

        # Check that we only have data for monitoring-app1
        assert (
            len(result) == 1
        ), f"Should have exactly 1 row for monitoring-app1, got {len(result)}"

        # Verify it's the correct application and metric
        assert (
            result[mm_schemas.WriterEvent.APPLICATION_NAME].iloc[0]
            == test_data[0]["application_name"]
        ), "Should only return monitoring-app1 data"
        assert (
            result[mm_schemas.MetricData.METRIC_NAME].iloc[0]
            == test_data[0]["metric_name"]
        ), "Should return accuracy metric"
        assert (
            abs(
                result[mm_schemas.MetricData.METRIC_VALUE].iloc[0]
                - test_data[0]["metric_value"]
            )
            < 0.001
        ), (
            f"Should return monitoring-app1's value ({test_data[0]['metric_value']}), "
            f"not monitoring-app2's value ({test_data[1]['metric_value']})"
        )

        # Query for monitoring-app2's accuracy metric
        test_metrics_app2 = [
            mm_schemas.ModelEndpointMonitoringMetric(
                project=query_test_helper.project_name,
                app="monitoring-app2",
                name="accuracy",
                type=mm_schemas.ModelEndpointMonitoringMetricType.METRIC,
            )
        ]

        result_app2 = metrics_handler.read_metrics_data_impl(
            endpoint_id="test_endpoint_1",
            start=datetime(2024, 1, 15, 0, 0, 0),
            end=datetime(2024, 1, 15, 23, 59, 59),
            metrics=test_metrics_app2,
        )

        # Verify we ONLY get monitoring-app2's data
        assert not result_app2.empty, "Should have returned data for monitoring-app2"
        assert (
            len(result_app2) == 1
        ), f"Should have exactly 1 row for monitoring-app2, got {len(result_app2)}"
        assert (
            result_app2[mm_schemas.WriterEvent.APPLICATION_NAME].iloc[0]
            == test_data[1]["application_name"]
        ), "Should only return monitoring-app2 data"
        assert (
            abs(
                result_app2[mm_schemas.MetricData.METRIC_VALUE].iloc[0]
                - test_data[1]["metric_value"]
            )
            < 0.001
        ), (
            f"Should return monitoring-app2's value ({test_data[1]['metric_value']}), "
            f"not monitoring-app1's value ({test_data[0]['metric_value']})"
        )
