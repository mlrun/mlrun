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


class TestResultsQueries:
    """Tests for results and metadata query operations."""

    def test_write_and_read_results_data(self, query_handler, sample_results):
        """Test writing and reading results data."""
        # Write test results first
        for result_data in sample_results:
            query_handler.write_application_event(
                result_data, mm_schemas.WriterEventKind.RESULT
            )

        # Now verify the data was actually written by reading it back
        connection = query_handler._connection
        app_results_table = query_handler.tables[
            mm_schemas.TimescaleDBTables.APP_RESULTS
        ]

        # Query the data back directly from the database
        query_result = connection.run(
            query=f"""
            SELECT endpoint_id, application_name, result_name, result_value, result_status, result_kind
            FROM {app_results_table.full_name()}
            WHERE endpoint_id = 'test_endpoint_1'
            ORDER BY end_infer_time DESC
            """
        )

        assert (
            len(query_result.data) == 1
        ), "Expected to find exactly 1 inserted result record"

        # Verify the data matches what we inserted
        row = query_result.data[0]
        (
            endpoint_id,
            application_name,
            result_name,
            result_value,
            result_status,
            result_kind,
        ) = row

        assert endpoint_id == "test_endpoint_1"
        assert application_name == "drift_app"
        assert result_name == "drift_detection"
        assert abs(result_value - 0.85) < 0.001  # Float comparison with tolerance
        assert result_status == mm_schemas.ResultStatusApp.detected.value
        assert result_kind == mm_schemas.ResultKindApp.concept_drift.value

        # Also test reading via metadata method to verify integration
        metadata_result = query_handler.get_results_metadata(
            endpoint_id="test_endpoint_1"
        )
        assert isinstance(metadata_result, pd.DataFrame)

        # Should have at least one row for our inserted data
        assert "endpoint_id" in metadata_result.columns
        test_endpoint_rows = metadata_result[
            metadata_result["endpoint_id"] == "test_endpoint_1"
        ]
        assert (
            len(test_endpoint_rows) == 1
        ), "Should find exactly 1 metadata record for the endpoint"

    def test_get_results_metadata(self, query_handler):
        """Test get_results_metadata method."""
        # First insert some results data to ensure we have metadata
        test_results = [
            {
                mm_schemas.WriterEvent.END_INFER_TIME: datetime(2024, 1, 15, 12, 0, 0),
                mm_schemas.WriterEvent.START_INFER_TIME: datetime(
                    2024, 1, 15, 12, 0, 0
                ),
                mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint",
                mm_schemas.WriterEvent.APPLICATION_NAME: "drift_app",
                mm_schemas.ResultData.RESULT_NAME: "drift_detection",
                mm_schemas.ResultData.RESULT_VALUE: 0.85,
                mm_schemas.ResultData.RESULT_STATUS: mm_schemas.ResultStatusApp.detected.value,
                mm_schemas.ResultData.RESULT_KIND: mm_schemas.ResultKindApp.concept_drift.value,
            },
            {
                mm_schemas.WriterEvent.END_INFER_TIME: datetime(2024, 1, 15, 12, 10, 0),
                mm_schemas.WriterEvent.START_INFER_TIME: datetime(
                    2024, 1, 15, 12, 10, 0
                ),
                mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint",
                mm_schemas.WriterEvent.APPLICATION_NAME: "performance_app",
                mm_schemas.ResultData.RESULT_NAME: "accuracy_check",
                mm_schemas.ResultData.RESULT_VALUE: 0.92,
                mm_schemas.ResultData.RESULT_STATUS: mm_schemas.ResultStatusApp.no_detection.value,
                mm_schemas.ResultData.RESULT_KIND: mm_schemas.ResultKindApp.model_performance.value,
            },
        ]

        for result_data in test_results:
            query_handler.write_application_event(
                result_data, mm_schemas.WriterEventKind.RESULT
            )

        result = query_handler.get_results_metadata(endpoint_id="test_endpoint")

        assert isinstance(result, pd.DataFrame)

        # Should have result_name column and verify our test results appear
        assert "result_name" in result.columns
        result_names = result["result_name"].unique()
        assert len(result_names) == 2  # We inserted 2 unique result names
        # Should include the results we inserted
        expected_names = ["drift_detection", "accuracy_check"]
        found_names = [name for name in expected_names if name in result_names]
        assert (
            len(found_names) == 2
        ), f"Expected to find both {expected_names} in {list(result_names)}"

        # Should have endpoint_id column and verify it matches our query
        assert "endpoint_id" in result.columns
        endpoints = result["endpoint_id"].unique()
        assert "test_endpoint" in endpoints

        # Should have application_name column and verify our test applications appear
        assert "application_name" in result.columns
        app_names = result["application_name"].unique()
        assert len(app_names) == 2  # We inserted 2 unique application names
        expected_apps = ["drift_app", "performance_app"]
        found_apps = [app for app in expected_apps if app in app_names]
        assert (
            len(found_apps) == 2
        ), f"Expected to find both {expected_apps} in {list(app_names)}"

    def test_count_results_by_status(self, query_handler):
        """Test count_results_by_status method."""
        connection = query_handler._connection
        app_results_table = query_handler.tables[
            mm_schemas.TimescaleDBTables.APP_RESULTS
        ]

        # Insert test data with different statuses and applications
        test_data = [
            (
                "endpoint_1",
                "drift_app",
                mm_schemas.ResultStatusApp.no_detection,
                datetime(2024, 1, 15, 12, 0, 0),
            ),  # Status: no_detection
            (
                "endpoint_1",
                "drift_app",
                mm_schemas.ResultStatusApp.potential_detection,
                datetime(2024, 1, 15, 12, 5, 0),
            ),  # Status: potential_detection
            (
                "endpoint_2",
                "drift_app",
                mm_schemas.ResultStatusApp.detected,
                datetime(2024, 1, 15, 12, 10, 0),
            ),  # Status: detected
            (
                "endpoint_2",
                "drift_app",
                mm_schemas.ResultStatusApp.potential_detection,
                datetime(2024, 1, 15, 12, 15, 0),
            ),  # Status: potential_detection
            (
                "endpoint_3",
                "performance_app",
                mm_schemas.ResultStatusApp.detected,
                datetime(2024, 1, 15, 12, 20, 0),
            ),  # Different app
        ]

        for endpoint_id, app_name, status, test_time in test_data:
            connection.run(
                statements=[
                    f"""
                    INSERT INTO {app_results_table.full_name()}
                    (end_infer_time, start_infer_time, endpoint_id, application_name, result_name,
                     result_value, result_status, result_kind, result_extra_data)
                    VALUES ('{test_time}', '{test_time}', '{endpoint_id}', '{app_name}', 'test_result',
                            0.85, {status}, {mm_schemas.ResultKindApp.concept_drift.value}, '{{}}')
                    """
                ]
            )

        result = query_handler.count_results_by_status(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
        )

        assert isinstance(result, dict)
        assert (
            len(result) == 4
        )  # Should have 4 distinct (app_name, status) combinations

        # Verify the structure: keys should be (app_name, status) tuples, values should be counts
        for key, count in result.items():
            assert isinstance(key, tuple)
            assert len(key) == 2  # (app_name, status)
            app_name, status = key
            assert isinstance(app_name, str)
            assert isinstance(status, int)  # Status should be integer value
            assert isinstance(count, int)
            assert count in [1, 2]  # Based on our test data, counts should be 1 or 2

        # Verify specific counts based on our test data
        expected_results = {
            ("drift_app", mm_schemas.ResultStatusApp.no_detection.value): 1,
            (
                "drift_app",
                mm_schemas.ResultStatusApp.potential_detection.value,
            ): 2,  # 2 records
            ("drift_app", mm_schemas.ResultStatusApp.detected.value): 1,
            ("performance_app", mm_schemas.ResultStatusApp.detected.value): 1,
        }

        for expected_key, expected_count in expected_results.items():
            assert (
                expected_key in result
            ), f"Expected key {expected_key} should be in result"
            assert (
                result[expected_key] == expected_count
            ), f"Expected {expected_count} for {expected_key}, got {result[expected_key]}"

    def test_get_drift_data(self, query_handler):
        """Test get_drift_data method with comprehensive drift scenarios."""
        connection = query_handler._connection
        app_results_table = query_handler.tables[
            mm_schemas.TimescaleDBTables.APP_RESULTS
        ]

        # Insert drift data with different statuses and time intervals
        test_data = [
            # Hour 1: 12:00 - 12:59
            (
                "endpoint_1",
                mm_schemas.ResultStatusApp.potential_detection,
                datetime(2024, 1, 15, 12, 0, 0),
            ),
            (
                "endpoint_2",
                mm_schemas.ResultStatusApp.detected,
                datetime(2024, 1, 15, 12, 15, 0),
            ),
            (
                "endpoint_1",
                mm_schemas.ResultStatusApp.detected,
                datetime(2024, 1, 15, 12, 30, 0),
            ),  # Later, higher status
            # Hour 2: 13:00 - 13:59
            (
                "endpoint_3",
                mm_schemas.ResultStatusApp.potential_detection,
                datetime(2024, 1, 15, 13, 0, 0),
            ),
            (
                "endpoint_4",
                mm_schemas.ResultStatusApp.detected,
                datetime(2024, 1, 15, 13, 30, 0),
            ),
            # Include no_detection to verify it's filtered out (should only count potential_detection=1 and detected=2)
            (
                "endpoint_5",
                mm_schemas.ResultStatusApp.no_detection,
                datetime(2024, 1, 15, 12, 45, 0),
            ),
        ]

        for endpoint_id, status, test_time in test_data:
            connection.run(
                statements=[
                    f"""
                    INSERT INTO {app_results_table.full_name()}
                    (end_infer_time, start_infer_time, endpoint_id, application_name, result_name,
                     result_value, result_status, result_kind, result_extra_data)
                    VALUES ('{test_time}', '{test_time}', '{endpoint_id}', 'drift_app', 'drift_result',
                            0.85, {status}, {mm_schemas.ResultKindApp.concept_drift.value}, '{{}}')
                    """
                ]
            )

        result = query_handler.get_drift_data(
            start=datetime(2024, 1, 15, 12, 0, 0),
            end=datetime(2024, 1, 15, 14, 0, 0),
            interval="1h",  # 1 hour intervals
        )

        # The result should be a ModelEndpointDriftValues object
        assert isinstance(
            result.values, list
        ), "Expected drift result to have values list"
        assert (
            len(result.values) == 2
        )  # Expected 2 hourly bins (12:00-13:00 and 13:00-14:00)

        # Verify the data structure contains _DriftBin objects
        for drift_bin in result.values:
            assert (
                drift_bin.timestamp is not None
            ), "Expected drift_bin to have timestamp"
            assert isinstance(
                drift_bin.count_suspected, int
            ), "Expected count_suspected to be int"
            assert isinstance(
                drift_bin.count_detected, int
            ), "Expected count_detected to be int"

            # Verify timestamp is within our query range (handle timezone-aware timestamps)
            start_time = datetime(2024, 1, 15, 12, 0, 0)
            end_time = datetime(2024, 1, 15, 14, 0, 0)

            # Convert to timezone-aware if the timestamp has timezone info
            if (
                hasattr(drift_bin.timestamp, "tzinfo")
                and drift_bin.timestamp.tzinfo is not None
            ):
                from datetime import timezone

                start_time = start_time.replace(tzinfo=timezone.utc)
                end_time = end_time.replace(tzinfo=timezone.utc)

            # Check if timestamp is a pandas Timestamp and convert for comparison
            timestamp_value = drift_bin.timestamp
            if hasattr(timestamp_value, "to_pydatetime"):
                timestamp_value = timestamp_value.to_pydatetime()

            assert timestamp_value >= start_time
            assert timestamp_value <= end_time

            # Counts should be non-negative integers
            assert isinstance(drift_bin.count_suspected, int)
            assert isinstance(drift_bin.count_detected, int)

        # Verify specific drift counts based on MAX aggregation logic
        # Hour 1 (12:00-13:00): endpoint_1->detected(2), endpoint_2->detected(2) = 2 detected, 0 suspected
        # Hour 2 (13:00-14:00): endpoint_3->potential(1), endpoint_4->detected(2) = 1 detected, 1 suspected
        expected_total_suspected = 1  # endpoint_3 in hour 2
        expected_total_detected = (
            3  # endpoint_1,endpoint_2 in hour 1 + endpoint_4 in hour 2
        )

        actual_total_suspected = sum(bin.count_suspected for bin in result.values)
        actual_total_detected = sum(bin.count_detected for bin in result.values)

        assert (
            actual_total_suspected == expected_total_suspected
        ), f"Expected {expected_total_suspected} suspected, got {actual_total_suspected}"
        assert (
            actual_total_detected == expected_total_detected
        ), f"Expected {expected_total_detected} detected, got {actual_total_detected}"
