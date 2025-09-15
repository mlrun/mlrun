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

import time
from datetime import datetime, timedelta

import pandas as pd

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.utils


class TestAggregationQueries:
    """Tests for aggregation query operations."""

    def test_get_avg_latency_raw_query(self, query_test_helper_with_aggregates):
        """Test get_avg_latency without interval (raw query path)."""
        connection = query_test_helper_with_aggregates.connection
        predictions_table = query_test_helper_with_aggregates.table_schemas[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]

        # Use recent timestamps for continuous aggregates to work properly
        now = mlrun.utils.datetime_now()
        base_time = now - timedelta(hours=2)  # 2 hours ago

        # Insert data spanning multiple time intervals with known latencies
        test_data = [
            # Hour 1: base_time (2 hours ago) - latency 0.1
            (base_time, "test_endpoint", 0.1),
            # Hour 2: base_time + 1h 15min and 45min - latencies 0.2 and 0.3 (avg should be 0.25)
            (base_time + timedelta(hours=1, minutes=15), "test_endpoint", 0.2),
            (base_time + timedelta(hours=1, minutes=45), "test_endpoint", 0.3),
            # Different endpoint for comparison
            (base_time + timedelta(hours=1, minutes=30), "other_endpoint", 0.5),
        ]

        for test_time, endpoint_id, latency in test_data:
            connection.run(
                statements=[
                    f"""
                    INSERT INTO {predictions_table.full_name()}
                    (end_infer_time, endpoint_id, latency, custom_metrics,
                     estimated_prediction_count, effective_sample_count)
                    VALUES ('{test_time}', '{endpoint_id}', {latency}, '{{}}', 1.0, 1)
                    """
                ]
            )

        # Create predictions handler using test helper
        predictions_handler = (
            query_test_helper_with_aggregates.create_predictions_handler()
        )

        # Test raw query (no interval specified)
        result = predictions_handler.get_avg_latency(
            endpoint_ids=["test_endpoint"],
            start=base_time - timedelta(minutes=30),  # Start slightly before our data
            end=now,  # End at current time
        )

        assert isinstance(result, pd.DataFrame)

        # Verify exact columns returned by raw data query
        assert "endpoint_id" in result.columns
        assert "avg_latency" in result.columns

        # Should have data for test_endpoint
        endpoint_results = result[result["endpoint_id"] == "test_endpoint"]
        assert len(endpoint_results) == 1

        # Verify concrete latency calculations
        # Expected: (0.1 + 0.2 + 0.3) / 3 = 0.2 (average of all test_endpoint data)
        avg_latency = endpoint_results["avg_latency"].iloc[0]
        assert abs(avg_latency - 0.2) < 0.01  # Allow small floating point variance

        # Test with multiple endpoints
        result_multi = predictions_handler.get_avg_latency(
            endpoint_ids=["test_endpoint", "other_endpoint"],
            start=base_time - timedelta(minutes=30),  # Start slightly before our data
            end=now,  # End at current time
        )

        assert len(result_multi) == 2  # Two endpoints

        # Verify other_endpoint has its exact latency
        other_results = result_multi[result_multi["endpoint_id"] == "other_endpoint"]
        assert len(other_results) == 1
        assert other_results["avg_latency"].iloc[0] == 0.5

    def test_get_avg_latency_with_pre_aggregate_interval(
        self, query_test_helper_with_aggregates, admin_connection
    ):
        """Test get_avg_latency WITH interval parameter for pre-aggregate optimization."""
        connection = query_test_helper_with_aggregates.connection
        predictions_table = query_test_helper_with_aggregates.table_schemas[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]

        # Create predictions handler using test helper
        predictions_handler = (
            query_test_helper_with_aggregates.create_predictions_handler()
        )

        # Use recent timestamps for continuous aggregates to work properly
        now = mlrun.utils.datetime_now()
        base_time = now - timedelta(hours=2)  # 2 hours ago

        # Insert data spanning multiple time intervals with known latencies
        test_data = [
            # Hour 1: base_time (2 hours ago) - latency 0.1
            (base_time, "test_endpoint", 0.1),
            # Hour 2: base_time + 1h 15min and 45min - latencies 0.2 and 0.3 (avg should be 0.25)
            (base_time + timedelta(hours=1, minutes=15), "test_endpoint", 0.2),
            (base_time + timedelta(hours=1, minutes=45), "test_endpoint", 0.3),
            # Different endpoint for comparison
            (base_time + timedelta(hours=1, minutes=30), "other_endpoint", 0.5),
        ]

        for test_time, endpoint_id, latency in test_data:
            connection.run(
                statements=[
                    f"""
                    INSERT INTO {predictions_table.full_name()}
                    (end_infer_time, endpoint_id, latency, custom_metrics,
                     estimated_prediction_count, effective_sample_count)
                    VALUES ('{test_time}', '{endpoint_id}', {latency}, '{{}}', 1.0, 1)
                    """
                ]
            )

        # Force refresh continuous aggregates for this table and interval

        # Force refresh the continuous aggregate and wait for it to process
        cagg_name = f"{predictions_table.full_name()}_cagg_1h"
        admin_connection.run(
            statements=[
                f"CALL refresh_continuous_aggregate('{cagg_name}', NULL, NULL);"
            ]
        )

        # Wait a moment for the refresh to complete
        time.sleep(0.1)

        # Verify the continuous aggregate has data by querying it directly
        check_query = f"""
        SELECT COUNT(*) as row_count
        FROM {cagg_name}
        WHERE endpoint_id = 'test_endpoint'
        """
        check_result = admin_connection.run(query=check_query)
        row_count = check_result.data[0][0] if check_result and check_result.data else 0

        # If no data in continuous aggregate, something is wrong with the test setup
        assert (
            row_count > 0
        ), f"Continuous aggregate {cagg_name} has no data for test_endpoint after refresh"

        # Test pre-aggregate query WITH interval parameter
        result = predictions_handler.get_avg_latency(
            endpoint_ids=["test_endpoint"],
            start=base_time - timedelta(minutes=30),  # Start slightly before our data
            end=now,  # End at current time
            interval="1h",  # Enable pre-aggregate optimization
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1  # Must have results (may be multiple time buckets)

        # Verify we have data for test_endpoint (pre-aggregates may have multiple time buckets)
        test_endpoint_data = result[result["endpoint_id"] == "test_endpoint"]
        assert len(test_endpoint_data) >= 1

        # Verify we have the expected pre-aggregate columns
        assert "time_bucket" in result.columns  # Pre-aggregate time column
        assert "avg_latency" in result.columns  # Aggregated latency
        assert "endpoint_id" in result.columns  # Endpoint identifier

        # Calculate overall average from all time buckets for test_endpoint
        # Each bucket may have different averages, but overall should be around 0.2
        overall_avg = test_endpoint_data["avg_latency"].mean()
        assert (
            abs(overall_avg - 0.2) < 0.05
        )  # Allow for pre-aggregate precision differences

    def test_get_drift_status_raw_query(self, query_test_helper_with_aggregates):
        """Test get_drift_status without interval (raw query path)."""
        connection = query_test_helper_with_aggregates.connection
        app_results_table = query_test_helper_with_aggregates.table_schemas[
            mm_schemas.TimescaleDBTables.APP_RESULTS
        ]

        # Use recent timestamp for continuous aggregates to work properly
        now = mlrun.utils.datetime_now()
        test_time = now - timedelta(minutes=30)  # 30 minutes ago
        connection.run(
            statements=[
                f"""
                INSERT INTO {app_results_table.full_name()}
                (end_infer_time, start_infer_time, endpoint_id, application_name, result_name,
                 result_value, result_status, result_kind, result_extra_data)
                VALUES ('{test_time}', '{test_time}', 'test_endpoint', 'drift_app', 'drift_result',
                        0.85, {mm_schemas.ResultStatusApp.detected.value},
                        {mm_schemas.ResultKindApp.concept_drift.value}, '{{}}')
                """
            ]
        )

        # Test raw query (no interval specified)
        # Create results handler using test helper
        results_handler = query_test_helper_with_aggregates.create_results_handler()

        result = results_handler.get_drift_status(
            endpoint_ids=["test_endpoint"],
            start=test_time - timedelta(minutes=10),  # Start slightly before our data
            end=now,  # End at current time
        )

        assert isinstance(result, pd.DataFrame)
        assert "endpoint_id" in result.columns

        # Should have exactly one row for our test endpoint
        assert len(result) == 1
        assert result["endpoint_id"].iloc[0] == "test_endpoint"

        # Verify the exact expected columns
        assert "result_status" in result.columns
        assert (
            result["result_status"].iloc[0] == mm_schemas.ResultStatusApp.detected.value
        )

    def test_get_drift_status_with_pre_aggregate_interval(
        self, query_test_helper_with_aggregates, admin_connection
    ):
        """Test get_drift_status WITH interval parameter for pre-aggregate optimization."""
        connection = query_test_helper_with_aggregates.connection
        app_results_table = query_test_helper_with_aggregates.table_schemas[
            mm_schemas.TimescaleDBTables.APP_RESULTS
        ]

        # Use hour-aligned timestamp for 1h continuous aggregates to work properly
        now = mlrun.utils.datetime_now()
        # Align to hour boundary for 1h continuous aggregate window calculation
        test_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)

        # Insert multiple drift records with different statuses
        test_data = [
            (
                test_time,
                "test_endpoint",
                mm_schemas.ResultStatusApp.detected.value,
                0.85,
            ),
            (
                test_time + timedelta(minutes=10),
                "test_endpoint",
                mm_schemas.ResultStatusApp.potential_detection.value,
                0.75,
            ),
            (
                test_time + timedelta(minutes=20),
                "other_endpoint",
                mm_schemas.ResultStatusApp.no_detection.value,
                0.15,
            ),
        ]

        for time_val, endpoint_id, status, result_value in test_data:
            connection.run(
                statements=[
                    f"""
                    INSERT INTO {app_results_table.full_name()}
                    (end_infer_time, start_infer_time, endpoint_id, application_name, result_name,
                     result_value, result_status, result_kind, result_extra_data)
                    VALUES ('{time_val}', '{time_val}', '{endpoint_id}', 'drift_app', 'drift_result',
                            {result_value}, {status}, {mm_schemas.ResultKindApp.concept_drift.value}, '{{}}')
                    """
                ]
            )

        # Force refresh continuous aggregates for this table and interval

        # Force refresh the continuous aggregate and wait for it to process
        cagg_name = f"{app_results_table.full_name()}_cagg_1h"
        admin_connection.run(
            statements=[
                f"CALL refresh_continuous_aggregate('{cagg_name}', NULL, NULL);"
            ]
        )

        # Wait a moment for the refresh to complete
        time.sleep(0.1)

        # Verify the continuous aggregate has data by querying it directly
        check_query = f"""
        SELECT COUNT(*) as row_count
        FROM {cagg_name}
        WHERE endpoint_id = 'test_endpoint'
        """
        check_result = admin_connection.run(query=check_query)
        row_count = check_result.data[0][0] if check_result and check_result.data else 0

        # If no data in continuous aggregate, something is wrong with the test setup
        assert (
            row_count > 0
        ), f"Continuous aggregate {cagg_name} has no data for test_endpoint after refresh"

        # Test pre-aggregate query WITH interval parameter
        # Create results handler using test helper
        results_handler = query_test_helper_with_aggregates.create_results_handler()

        result = results_handler.get_drift_status(
            endpoint_ids=["test_endpoint", "other_endpoint"],
            start=test_time
            - timedelta(hours=1),  # Start 1 hour before to include the hour bucket
            end=now,  # End at current time
            interval="1h",  # Enable pre-aggregate optimization
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1  # Must have results (may be multiple time buckets)

        # Verify we have data for at least one of the requested endpoints
        # (pre-aggregates may filter based on max/latest values per time bucket)
        endpoint_values = set(result["endpoint_id"].values)
        assert endpoint_values.intersection({"test_endpoint", "other_endpoint"})

        # Verify we have the expected pre-aggregate columns
        assert "time_bucket" in result.columns  # Pre-aggregate time column
        assert "endpoint_id" in result.columns  # Endpoint identifier
        assert "application_name" in result.columns  # Application name
        assert "result_name" in result.columns  # Result name

        # Verify we have meaningful data in the result
        assert result["application_name"].iloc[0] == "drift_app"
        assert result["result_name"].iloc[0] == "drift_result"


class TestPreAggregateExceptionHandling:
    """Tests for pre-aggregate exception handling and fallback behavior."""

    def test_get_avg_latency_pre_aggregate_exception_fallback(
        self, query_test_helper_with_aggregates
    ):
        """Test that get_avg_latency properly falls back to raw query when pre-aggregates fail."""
        connection = query_test_helper_with_aggregates.connection
        predictions_table = query_test_helper_with_aggregates.table_schemas[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]

        # Create predictions handler using test helper
        predictions_handler = (
            query_test_helper_with_aggregates.create_predictions_handler()
        )

        # Use recent timestamps
        now = mlrun.utils.datetime_now()
        base_time = now - timedelta(hours=1)

        # Insert test data
        connection.run(
            statements=[
                f"""
                INSERT INTO {predictions_table.full_name()}
                (end_infer_time, endpoint_id, latency, custom_metrics,
                 estimated_prediction_count, effective_sample_count)
                VALUES ('{base_time}', 'test_endpoint', 0.15, '{{}}', 1.0, 1)
                """
            ]
        )

        # Test with interval that might not have pre-aggregates available
        # This should trigger fallback to raw query
        result = predictions_handler.get_avg_latency(
            endpoint_ids=["test_endpoint"],
            start=base_time - timedelta(minutes=10),
            end=now,
            interval="15m",  # Non-standard interval that may not have pre-aggregates
        )

        # Should still get valid results via fallback
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # Exactly one endpoint

        # Verify exact values
        assert result["endpoint_id"].iloc[0] == "test_endpoint"
        assert result["avg_latency"].iloc[0] == 0.15  # Exact latency we inserted

    def test_get_drift_status_pre_aggregate_exception_fallback(
        self, query_test_helper_with_aggregates
    ):
        """Test that get_drift_status properly falls back to raw query when pre-aggregates fail."""
        connection = query_test_helper_with_aggregates.connection
        app_results_table = query_test_helper_with_aggregates.table_schemas[
            mm_schemas.TimescaleDBTables.APP_RESULTS
        ]

        now = mlrun.utils.datetime_now()
        test_time = now - timedelta(minutes=30)

        connection.run(
            statements=[
                f"""
                INSERT INTO {app_results_table.full_name()}
                (end_infer_time, start_infer_time, endpoint_id, application_name, result_name,
                 result_value, result_status, result_kind, result_extra_data)
                VALUES ('{test_time}', '{test_time}', 'test_endpoint', 'drift_app', 'drift_result',
                        0.85, {mm_schemas.ResultStatusApp.detected.value},
                        {mm_schemas.ResultKindApp.concept_drift.value}, '{{}}')
                """
            ]
        )

        # Test with interval that might not have pre-aggregates available
        # Create results handler using test helper
        results_handler = query_test_helper_with_aggregates.create_results_handler()

        result = results_handler.get_drift_status(
            endpoint_ids=["test_endpoint"],
            start=test_time - timedelta(minutes=10),
            end=now,
            interval="15m",  # Non-standard interval that may not have pre-aggregates
        )

        # Should still get valid results via fallback
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # Exactly one endpoint

        # Verify exact values
        assert result["endpoint_id"].iloc[0] == "test_endpoint"
        assert (
            result["result_status"].iloc[0] == mm_schemas.ResultStatusApp.detected.value
        )


class TestLatestMetricsCalculation:
    """Tests for calculate_latest_metrics method."""

    def test_calculate_latest_metrics_empty_application_list(self, query_test_helper):
        """Test calculate_latest_metrics with empty application list."""
        # Create metrics handler using test helper
        metrics_handler = query_test_helper.create_metrics_handler()

        result = metrics_handler.calculate_latest_metrics(
            application_names=[],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        assert isinstance(result, list)
        assert len(result) == 0

    def test_calculate_latest_metrics_no_data(self, query_test_helper):
        """Test calculate_latest_metrics with applications that have no data."""
        # Create metrics handler using test helper
        metrics_handler = query_test_helper.create_metrics_handler()

        application_names = ["nonexistent_1", "nonexistent_2", "nonexistent_3"]

        result = metrics_handler.calculate_latest_metrics(
            application_names=application_names,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        assert isinstance(result, list)
        assert len(result) == 0  # No data for nonexistent applications

    def test_calculate_latest_metrics_with_drift_data_only(self, query_test_helper):
        """Test calculate_latest_metrics with only drift/result data."""
        # Create metrics handler using test helper
        metrics_handler = query_test_helper.create_metrics_handler()

        connection = query_test_helper.connection
        app_results_table = query_test_helper.table_schemas[
            mm_schemas.TimescaleDBTables.APP_RESULTS
        ]

        # Insert drift data for multiple endpoints with different statuses
        test_data = [
            (
                "endpoint_1",
                mm_schemas.ResultStatusApp.potential_detection,
                datetime(2024, 1, 15, 12, 0, 0),
                0.75,
            ),  # Status: potential_detection
            (
                "endpoint_1",
                mm_schemas.ResultStatusApp.detected,
                datetime(2024, 1, 15, 12, 10, 0),
                0.85,
            ),  # Later timestamp, higher status: detected - should be the latest
            (
                "endpoint_2",
                mm_schemas.ResultStatusApp.detected,
                datetime(2024, 1, 15, 12, 5, 0),
                0.9,
            ),  # Status: detected
            (
                "endpoint_3",
                mm_schemas.ResultStatusApp.potential_detection,
                datetime(2024, 1, 15, 12, 15, 0),
                0.6,
            ),  # Status: potential_detection
        ]

        for endpoint_id, status, test_time, result_value in test_data:
            connection.run(
                statements=[
                    f"""
                    INSERT INTO {app_results_table.full_name()}
                    (end_infer_time, start_infer_time, endpoint_id, application_name, result_name,
                     result_value, result_status, result_kind, result_extra_data)
                    VALUES ('{test_time}', '{test_time}', '{endpoint_id}', 'drift_app', 'drift_result',
                            {result_value}, {status}, {mm_schemas.ResultKindApp.concept_drift.value}, '{{}}')
                    """
                ]
            )

        result = metrics_handler.calculate_latest_metrics(
            application_names=["drift_app"],
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
        )

        assert isinstance(result, list)
        assert (
            len(result) == 1
        )  # DISTINCT ON (result_name) returns 1 record for 'drift_result'

        # Verify the exact expected result (latest by timestamp: endpoint_3 at 12:15:00)
        from datetime import timezone

        expected_result = mm_schemas.ApplicationResultRecord(
            time=datetime(2024, 1, 15, 12, 15, 0, tzinfo=timezone.utc),
            value=0.6,
            kind=mm_schemas.ResultKindApp.concept_drift,
            status=mm_schemas.ResultStatusApp.potential_detection,
            result_name="drift_result",
        )

        assert result[0] == expected_result


class TestEndpointCounting:
    """Tests for endpoint counting operations."""

    def test_count_processed_model_endpoints_no_data(self, query_test_helper):
        """Test count_processed_model_endpoints with no data."""
        # Create predictions handler using test helper
        predictions_handler = query_test_helper.create_predictions_handler()

        result = predictions_handler.count_processed_model_endpoints(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            application_names=["test_app"],  # Test application name
        )

        assert result.get("test_app", 0) == 0  # No data for test application

    def test_count_processed_model_endpoints_with_data(self, query_test_helper):
        """Test count_processed_model_endpoints with sample data and application linkage."""
        # Create predictions handler using test helper
        predictions_handler = query_test_helper.create_predictions_handler()

        connection = query_test_helper.connection
        predictions_table = query_test_helper.table_schemas[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]
        metrics_table = query_test_helper.table_schemas[
            mm_schemas.TimescaleDBTables.METRICS
        ]

        # Insert prediction data for multiple unique endpoints
        predictions_data = [
            ("endpoint_1", datetime(2024, 1, 15, 12, 0, 0)),
            ("endpoint_2", datetime(2024, 1, 15, 12, 5, 0)),
            (
                "endpoint_1",
                datetime(2024, 1, 15, 12, 10, 0),
            ),  # Duplicate endpoint - should count as 1
            ("endpoint_3", datetime(2024, 1, 15, 12, 15, 0)),
        ]

        for endpoint_id, test_time in predictions_data:
            connection.run(
                statements=[
                    f"""
                    INSERT INTO {predictions_table.full_name()}
                    (end_infer_time, endpoint_id, latency, custom_metrics,
                     estimated_prediction_count, effective_sample_count)
                    VALUES ('{test_time}', '{endpoint_id}', 0.1, '{{}}', 1.0, 1)
                    """
                ]
            )

        # Insert metrics data to link endpoints to applications (needed for JOIN query)
        metrics_data = [
            ("endpoint_1", "test_app", datetime(2024, 1, 15, 12, 0, 0)),
            ("endpoint_2", "test_app", datetime(2024, 1, 15, 12, 5, 0)),
            (
                "endpoint_3",
                "other_app",
                datetime(2024, 1, 15, 12, 15, 0),
            ),  # Different app
        ]

        for endpoint_id, app_name, test_time in metrics_data:
            connection.run(
                statements=[
                    f"""
                    INSERT INTO {metrics_table.full_name()}
                    (end_infer_time, start_infer_time, endpoint_id, application_name, metric_name, metric_value)
                    VALUES ('{test_time}', '{test_time}', '{endpoint_id}', '{app_name}', 'test_metric', 0.95)
                    """
                ]
            )

        result = predictions_handler.count_processed_model_endpoints(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            application_names=["test_app"],  # Test application name
        )

        # Should now have 2 unique endpoints (endpoint_1 and endpoint_2) for test_app
        assert (
            result["test_app"] == 2
        )  # endpoint_1 and endpoint_2 are linked to test_app

        # Test with different application
        result_other = predictions_handler.count_processed_model_endpoints(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            application_names=["other_app"],
        )

        assert result_other["other_app"] == 1  # Only endpoint_3 is linked to other_app

    def test_count_processed_model_endpoints_time_filtering(self, query_test_helper):
        """Test that count_processed_model_endpoints respects time range."""
        # Create predictions handler using test helper
        predictions_handler = query_test_helper.create_predictions_handler()

        connection = query_test_helper.connection
        predictions_table = query_test_helper.table_schemas[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]

        # Insert data both inside and outside the query time range
        endpoints_data = [
            ("endpoint_1", datetime(2024, 1, 14, 12, 0, 0)),  # Before range
            ("endpoint_2", datetime(2024, 1, 15, 12, 0, 0)),  # In range
            ("endpoint_3", datetime(2024, 1, 15, 12, 5, 0)),  # In range
            ("endpoint_4", datetime(2024, 1, 17, 12, 0, 0)),  # After range
        ]

        for endpoint_id, test_time in endpoints_data:
            connection.run(
                statements=[
                    f"""
                    INSERT INTO {predictions_table.full_name()}
                    (end_infer_time, endpoint_id, latency, custom_metrics,
                     estimated_prediction_count, effective_sample_count)
                    VALUES ('{test_time}', '{endpoint_id}', 0.1, '{{}}', 1.0, 1)
                    """
                ]
            )

        # Insert metrics data to link endpoints to application (required for JOIN)
        metrics_table = query_test_helper.table_schemas[
            mm_schemas.TimescaleDBTables.METRICS
        ]

        # Link only the endpoints within time range to test_app
        metrics_data = [
            ("endpoint_2", datetime(2024, 1, 15, 12, 0, 0)),  # In range
            ("endpoint_3", datetime(2024, 1, 15, 12, 5, 0)),  # In range
        ]

        for endpoint_id, test_time in metrics_data:
            connection.run(
                statements=[
                    f"""
                    INSERT INTO {metrics_table.full_name()}
                    (end_infer_time, start_infer_time, endpoint_id, application_name, metric_name, metric_value)
                    VALUES ('{test_time}', '{test_time}', '{endpoint_id}', 'test_app', 'test_metric', 0.95)
                    """
                ]
            )

        # Query for specific time range (only 2024-01-15 to 2024-01-16)
        result = predictions_handler.count_processed_model_endpoints(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            application_names=["test_app"],
        )

        assert (
            result["test_app"] == 2
        )  # Only endpoint_2 and endpoint_3 are within time range and linked to test_app
