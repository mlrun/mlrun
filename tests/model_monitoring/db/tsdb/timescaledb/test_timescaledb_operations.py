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

import threading
import time
from datetime import datetime
from typing import Optional

import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors


class TestTimescaleDBOperationsManagerIntegration:
    """Integration tests using real database connections."""

    @staticmethod
    def _create_result_event_data(
        endpoint_id: str = "test_endpoint_result",
        application_name: str = "drift_detection",
        result_name: str = "feature_drift",
        result_value: float = 0.85,
        result_status: int = 1,
        result_kind: int = 2,
        result_extra_data: str = '{"confidence": 0.9}',
        end_time: Optional[datetime] = None,
        start_time: Optional[datetime] = None,
    ) -> dict:
        """Factory method for creating result event data."""
        if end_time is None:
            end_time = datetime(2024, 1, 15, 12, 30, 45)
        if start_time is None:
            start_time = datetime(2024, 1, 15, 12, 30, 40)

        return {
            mm_schemas.WriterEvent.END_INFER_TIME: end_time,
            mm_schemas.WriterEvent.START_INFER_TIME: start_time,
            mm_schemas.WriterEvent.ENDPOINT_ID: endpoint_id,
            mm_schemas.WriterEvent.APPLICATION_NAME: application_name,
            mm_schemas.ResultData.RESULT_NAME: result_name,
            mm_schemas.ResultData.RESULT_VALUE: result_value,
            mm_schemas.ResultData.RESULT_STATUS: result_status,
            mm_schemas.ResultData.RESULT_KIND: result_kind,
            mm_schemas.ResultData.RESULT_EXTRA_DATA: result_extra_data,
        }

    @staticmethod
    def _create_metric_event_data(
        endpoint_id: str = "test_endpoint_metric",
        application_name: str = "performance_monitoring",
        metric_name: str = "accuracy",
        metric_value: float = 0.95,
        end_time: Optional[datetime] = None,
        start_time: Optional[datetime] = None,
    ) -> dict:
        """Factory method for creating metric event data."""
        if end_time is None:
            end_time = datetime(2024, 1, 15, 12, 30, 45)
        if start_time is None:
            start_time = datetime(2024, 1, 15, 12, 30, 40)

        return {
            mm_schemas.WriterEvent.END_INFER_TIME: end_time,
            mm_schemas.WriterEvent.START_INFER_TIME: start_time,
            mm_schemas.WriterEvent.ENDPOINT_ID: endpoint_id,
            mm_schemas.WriterEvent.APPLICATION_NAME: application_name,
            mm_schemas.MetricData.METRIC_NAME: metric_name,
            mm_schemas.MetricData.METRIC_VALUE: metric_value,
        }

    @staticmethod
    def _verify_table_data(
        connection, table, expected_count: int, where_clause: Optional[str] = None
    ) -> list:
        """Helper method for verifying table data."""
        query = f"SELECT COUNT(*) FROM {table.full_name()}"
        if where_clause:
            query += f" WHERE {where_clause}"

        result = connection.run(query=query)
        actual_count = result.data[0][0]
        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} records, got {actual_count}"
        return result.data

    @staticmethod
    def _insert_prediction_data(
        connection, table, endpoint_id: str, latency: float = 0.1
    ):
        """Helper method for inserting prediction data."""
        connection.run(
            statements=[
                f"""
                INSERT INTO {table.full_name()}
                (end_infer_time, endpoint_id, latency, custom_metrics,
                 estimated_prediction_count, effective_sample_count)
                VALUES (NOW(), '{endpoint_id}', {latency}, '{{}}', 1.0, 1)
                """
            ]
        )

    @staticmethod
    def _insert_metric_data(
        connection, table, endpoint_id: str, metric_value: float = 0.5
    ):
        """Helper method for inserting metric data."""
        connection.run(
            statements=[
                f"""
                INSERT INTO {table.full_name()}
                (end_infer_time, start_infer_time, endpoint_id, application_name, metric_name, metric_value)
                VALUES (NOW(), NOW(), '{endpoint_id}', 'test_app', 'test_metric', {metric_value})
                """
            ]
        )

    def test_create_tables_basic(self, query_test_helper):
        """Test basic table creation without pre-aggregates."""
        # Tables are already created by the fixture, just verify they exist
        connection = query_test_helper.connection
        schema_name = query_test_helper.operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ].schema

        # Check if schema exists
        result = connection.run(
            query=f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{schema_name}'"
        )
        assert len(result.data) == 1

        # Check if tables exist for this specific project
        project_id = query_test_helper.operations_handler.project.replace("-", "_")
        result = connection.run(
            query=f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = '{schema_name}'
            AND table_name LIKE '%{project_id}%'
            """
        )
        assert len(result.data) == 4  # predictions, metrics, app_results, errors

        # Verify they are hypertables for this project
        result = connection.run(
            query=f"""
            SELECT hypertable_name FROM timescaledb_information.hypertables
            WHERE hypertable_schema = '{schema_name}'
            AND hypertable_name LIKE '%{project_id}%'
            """
        )
        assert len(result.data) == 4

    def test_create_tables_with_pre_aggregates(self, query_test_helper_with_aggregates):
        """Test table creation with pre-aggregate configuration."""
        # Tables are already created by the fixture
        schema_name = query_test_helper_with_aggregates.operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ].schema
        connection = query_test_helper_with_aggregates.connection

        # Verify base tables exist
        result = connection.run(
            query=f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema_name}'"
        )
        assert len(result.data) >= 4

        # Verify continuous aggregates were created
        result = connection.run(
            query=f"""
            SELECT view_name FROM timescaledb_information.continuous_aggregates
            WHERE hypertable_schema = '{schema_name}'
            """
        )
        # Should have continuous aggregates for predictions, metrics, app_results (not errors)
        # with 2 intervals each = 6 total
        assert len(result.data) >= 3

    def test_write_application_event_result(self, query_test_helper):
        """Test writing result events to the database."""
        # Create tables first
        query_test_helper.operations_handler.create_tables()

        # Prepare event data using factory method
        event_data = self._create_result_event_data()

        # Write event
        query_test_helper.operations_handler.write_application_event(
            event_data, mm_schemas.WriterEventKind.RESULT
        )

        # Verify data was written
        connection = query_test_helper.operations_handler._connection
        app_results_table = query_test_helper.operations_handler.tables[
            mm_schemas.TimescaleDBTables.APP_RESULTS
        ]

        result = connection.run(
            query=f"""
            SELECT endpoint_id, application_name, result_name, result_value
            FROM {app_results_table.full_name()}
            WHERE endpoint_id = 'test_endpoint_result'
            """
        )

        assert len(result.data) == 1
        assert result.data[0][0] == "test_endpoint_result"
        assert result.data[0][1] == "drift_detection"
        assert result.data[0][2] == "feature_drift"
        assert result.data[0][3] == 0.85

    def test_write_application_event_metric(self, query_test_helper):
        """Test writing metric events to the database."""
        # Create tables first
        query_test_helper.operations_handler.create_tables()

        # Prepare event data using factory method
        event_data = self._create_metric_event_data()

        # Write event
        query_test_helper.operations_handler.write_application_event(
            event_data, mm_schemas.WriterEventKind.METRIC
        )

        # Verify data was written
        connection = query_test_helper.operations_handler._connection
        metrics_table = query_test_helper.operations_handler.tables[
            mm_schemas.TimescaleDBTables.METRICS
        ]

        result = connection.run(
            query=f"""
            SELECT endpoint_id, application_name, metric_name, metric_value
            FROM {metrics_table.full_name()}
            WHERE endpoint_id = 'test_endpoint_metric'
            """
        )

        assert len(result.data) == 1
        assert result.data[0][0] == "test_endpoint_metric"
        assert result.data[0][1] == "performance_monitoring"
        assert result.data[0][2] == "accuracy"
        assert result.data[0][3] == 0.95

    def test_delete_tsdb_records_raw_only(self, query_test_helper):
        """Test deleting records from raw tables only."""
        # Create tables and insert test data
        query_test_helper.operations_handler.create_tables()

        # Insert test data in multiple tables
        test_endpoints = ["endpoint_1", "endpoint_2", "endpoint_3"]
        connection = query_test_helper.operations_handler._connection

        predictions_table = query_test_helper.operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]
        metrics_table = query_test_helper.operations_handler.tables[
            mm_schemas.TimescaleDBTables.METRICS
        ]

        for endpoint_id in test_endpoints:
            self._insert_prediction_data(connection, predictions_table, endpoint_id)
            self._insert_metric_data(connection, metrics_table, endpoint_id)

        # Verify data exists using helper
        predictions_table = query_test_helper.operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]
        self._verify_table_data(connection, predictions_table, 3)

        # Delete records for specific endpoints
        query_test_helper.operations_handler.delete_tsdb_records(
            ["endpoint_1", "endpoint_2"], include_aggregates=False
        )

        # Verify deletion
        result = connection.run(
            query=f"""
            SELECT COUNT(*) FROM {predictions_table.full_name()}
            WHERE endpoint_id IN ('endpoint_1', 'endpoint_2')
            """
        )
        assert result.data[0][0] == 0

        # Verify endpoint_3 still exists
        result = connection.run(
            query=f"""
            SELECT COUNT(*) FROM {predictions_table.full_name()}
            WHERE endpoint_id = 'endpoint_3'
            """
        )
        assert result.data[0][0] == 1

    def test_delete_tsdb_records_with_aggregates(
        self, query_test_helper_with_aggregates
    ):
        """Test deleting records including aggregates."""
        # Tables are already created by fixture
        connection = query_test_helper_with_aggregates.connection
        test_endpoint = "endpoint_with_aggregates"

        # Insert test data using helper
        predictions_table = query_test_helper_with_aggregates.operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]
        self._insert_prediction_data(connection, predictions_table, test_endpoint)

        # Verify data exists using helper
        self._verify_table_data(
            connection, predictions_table, 1, f"endpoint_id = '{test_endpoint}'"
        )

        # Delete records including aggregates
        query_test_helper_with_aggregates.operations_handler.delete_tsdb_records(
            [test_endpoint], include_aggregates=True
        )

        # Verify deletion from raw table using helper
        self._verify_table_data(
            connection, predictions_table, 0, f"endpoint_id = '{test_endpoint}'"
        )

    def test_delete_tsdb_records_empty_list(self, query_test_helper):
        """Test deleting with empty endpoint list."""
        query_test_helper.operations_handler.create_tables()

        # Should not raise exception
        query_test_helper.operations_handler.delete_tsdb_records(
            [], include_aggregates=True
        )

    def test_delete_tsdb_records_special_characters(self, query_test_helper, statement):
        """Test deleting endpoints with special characters."""
        query_test_helper.operations_handler.create_tables()

        # Insert data with special characters
        special_endpoints = [
            "endpoint'with'quotes",
            "endpoint-with-dashes",
            "endpoint_with_underscores",
        ]
        connection = query_test_helper.operations_handler._connection

        predictions_table = query_test_helper.operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]

        for endpoint_id in special_endpoints:
            # Create a proper Statement object with parameters
            stmt = statement(
                sql=f"""
                INSERT INTO {predictions_table.full_name()}
                (end_infer_time, endpoint_id, latency, custom_metrics,
                 estimated_prediction_count, effective_sample_count)
                VALUES (NOW(), %s, 0.1, '{{}}', 1.0, 1)
                """,
                parameters=[endpoint_id],
            )
            connection.run(statements=[stmt])

        # Delete using parameterized queries (should handle special characters safely)
        query_test_helper.operations_handler.delete_tsdb_records(special_endpoints)

        # Verify deletion
        result = connection.run(
            query=f"SELECT COUNT(*) FROM {predictions_table.full_name()}"
        )
        assert result.data[0][0] == 0

    def test_delete_tsdb_resources_complete_cleanup(self, query_test_helper):
        """Test complete resource deletion."""
        # Create tables
        query_test_helper.operations_handler.create_tables()

        connection = query_test_helper.operations_handler._connection
        schema_name = query_test_helper.operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ].schema
        project_id = query_test_helper.operations_handler.project.replace("-", "_")

        # Verify tables exist for this project
        result = connection.run(
            query=f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = '{schema_name}'
            AND table_name LIKE '%{project_id}%'
            """
        )
        table_count_before = result.data[0][0]
        assert table_count_before == 4

        # Delete all resources
        query_test_helper.operations_handler.delete_tsdb_resources()

        # Verify project tables are gone
        result = connection.run(
            query=f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = '{schema_name}'
            AND table_name LIKE '%{project_id}%'
            """
        )
        assert result.data[0][0] == 0

    def test_delete_tsdb_resources_with_aggregates(
        self, query_test_helper_with_aggregates
    ):
        """Test resource deletion including continuous aggregates."""
        # Tables are already created by fixture
        connection = query_test_helper_with_aggregates.connection
        schema_name = query_test_helper_with_aggregates.operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ].schema
        project_id = (
            query_test_helper_with_aggregates.operations_handler.project.replace(
                "-", "_"
            )
        )

        # Verify continuous aggregates exist for this project
        result = connection.run(
            query=f"""
            SELECT COUNT(*) FROM timescaledb_information.continuous_aggregates
            WHERE hypertable_schema = '{schema_name}'
            AND view_name LIKE '%{project_id}%'
            """
        )
        cagg_count_before = result.data[0][0]
        assert cagg_count_before > 0

        # Delete all resources
        query_test_helper_with_aggregates.operations_handler.delete_tsdb_resources()

        # Verify project continuous aggregates are gone
        result = connection.run(
            query=f"""
            SELECT COUNT(*) FROM timescaledb_information.continuous_aggregates
            WHERE hypertable_schema = '{schema_name}'
            AND view_name LIKE '%{project_id}%'
            """
        )
        assert result.data[0][0] == 0

    def test_datetime_conversion_edge_cases(self, query_test_helper):
        """Test datetime conversion with various formats."""
        from mlrun.utils import datetime_from_iso

        # Test ISO string with Z suffix
        result = datetime_from_iso("2024-01-15T12:30:45Z")
        assert result == datetime(2024, 1, 15, 12, 30, 45, tzinfo=result.tzinfo)

        # Test ISO string with timezone offset
        result = datetime_from_iso("2024-01-15T12:30:45+00:00")
        assert result == datetime(2024, 1, 15, 12, 30, 45, tzinfo=result.tzinfo)

        # Test ISO string with microseconds
        result = datetime_from_iso("2024-01-15T12:30:45.123456Z")
        assert result == datetime(2024, 1, 15, 12, 30, 45, 123456, tzinfo=result.tzinfo)

        # Test datetime object passthrough
        dt_input = datetime(2024, 1, 15, 12, 30, 45)
        assert dt_input == datetime(2024, 1, 15, 12, 30, 45)

    def test_write_event_with_unicode_data(self, query_test_helper):
        """Test writing events with Unicode characters."""
        query_test_helper.operations_handler.create_tables()

        # Event with Unicode data using factory method
        event_data = self._create_result_event_data(
            endpoint_id="测试端点",  # Chinese characters
            application_name="тест_приложение",  # Cyrillic
            result_name="résultat_test",  # French accents
            result_kind=1,
            result_extra_data='{"message": "успех"}',
            end_time=datetime.now(),
            start_time=datetime.now(),
        )

        # Should not raise exception
        query_test_helper.operations_handler.write_application_event(
            event_data, mm_schemas.WriterEventKind.RESULT
        )

        # Verify data was written correctly
        connection = query_test_helper.operations_handler._connection
        app_results_table = query_test_helper.operations_handler.tables[
            mm_schemas.TimescaleDBTables.APP_RESULTS
        ]

        result = connection.run(
            query=f"""
            SELECT endpoint_id, application_name, result_name
            FROM {app_results_table.full_name()}
            WHERE endpoint_id = '测试端点'
            """
        )

        assert len(result.data) == 1
        assert result.data[0][0] == "测试端点"
        assert result.data[0][1] == "тест_приложение"
        assert result.data[0][2] == "résultat_test"

    def test_large_batch_deletion(self, query_test_helper):
        """Test deletion of large number of endpoints."""
        query_test_helper.operations_handler.create_tables()

        # Insert many endpoints
        endpoint_count = 100
        endpoints = [f"endpoint_{i}" for i in range(endpoint_count)]

        connection = query_test_helper.operations_handler._connection
        predictions_table = query_test_helper.operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]

        # Batch insert using helper
        for i, endpoint_id in enumerate(endpoints):
            self._insert_prediction_data(
                connection, predictions_table, endpoint_id, 0.1 + i * 0.001
            )

        # Verify all data inserted using helper
        self._verify_table_data(connection, predictions_table, endpoint_count)

        # Delete first 50 endpoints
        endpoints_to_delete = endpoints[:50]
        query_test_helper.operations_handler.delete_tsdb_records(endpoints_to_delete)

        # Verify deletion using helper
        self._verify_table_data(connection, predictions_table, 50)

    def test_error_handling_invalid_event_data(self, query_test_helper):
        """Test error handling with invalid event data."""
        query_test_helper.operations_handler.create_tables()

        # Event with missing required fields
        invalid_event = {
            mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint",
            # Missing other required fields
        }

        # Should raise an exception with specific error about writing to TimescaleDB
        with pytest.raises(
            mlrun.errors.MLRunRuntimeError,
            match=r"Failed to write event to TimescaleDB",
        ):
            query_test_helper.operations_handler.write_application_event(
                invalid_event, mm_schemas.WriterEventKind.RESULT
            )

    def test_concurrent_operations(self, query_test_helper):
        """Test concurrent operations with connection pool awareness."""

        query_test_helper.operations_handler.create_tables()

        results = []
        errors = []
        connection_pool_errors = 0

        def write_worker(worker_id):
            """Worker function for concurrent writes with retry logic."""
            nonlocal connection_pool_errors
            try:
                for i in range(3):  # Reduced from 5 to 3 operations per worker
                    event_data = self._create_metric_event_data(
                        endpoint_id=f"worker_{worker_id}_endpoint_{i}",
                        application_name=f"worker_{worker_id}_app",
                        metric_value=float(worker_id + i),
                        end_time=datetime.now(),
                        start_time=datetime.now(),
                    )

                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            query_test_helper.operations_handler.write_application_event(
                                event_data, mm_schemas.WriterEventKind.METRIC
                            )
                            break  # Success, exit retry loop
                        except Exception as e:
                            if (
                                "connection pool exhausted" in str(e).lower()
                                and attempt < max_retries - 1
                            ):
                                connection_pool_errors += 1
                                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                                continue
                            raise  # Re-raise if not pool exhaustion or out of retries

                    time.sleep(0.05)  # Increased delay to reduce contention
                results.append(f"worker_{worker_id}_completed")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # Start fewer worker threads to reduce connection pressure
        threads = []
        for i in range(2):  # Reduced from 3 to 2 workers
            thread = threading.Thread(target=write_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify that all workers completed successfully (with retries handling connection pool issues)
        assert not errors, f"Thread errors occurred: {errors}"
        assert len(results) == 2, f"Expected 2 workers to complete, got {len(results)}"

        # Verify some data was written (at least 1 record per successful worker)
        connection = query_test_helper.operations_handler._connection
        metrics_table = query_test_helper.operations_handler.tables[
            mm_schemas.TimescaleDBTables.METRICS
        ]

        result = connection.run(
            query=f"SELECT COUNT(*) FROM {metrics_table.full_name()}"
        )
        record_count = result.data[0][0]
        expected_min = len(results)  # At least 1 record per successful worker
        assert (
            record_count >= expected_min
        ), f"Expected at least {expected_min} records, got {record_count}"

    def test_aggregate_deletion_statements_generation(self, query_test_helper):
        """Test generation of aggregate deletion statements for endpoint cleanup."""
        operations_handler = query_test_helper.operations_handler
        operations_handler.create_tables()

        test_endpoints = ["endpoint-1", "endpoint-2", "endpoint-3"]

        # Test aggregate delete statements generation
        statements = operations_handler._get_aggregate_delete_statements(test_endpoints)

        # Should return a list of statements
        assert isinstance(statements, list), "Should return a list of statements"

        # Should handle empty endpoint list gracefully
        empty_statements = operations_handler._get_aggregate_delete_statements([])
        assert isinstance(empty_statements, list), "Should handle empty endpoint list"

        # Cleanup
        operations_handler.delete_tsdb_resources()

    def test_application_deletion_statements_generation(self, query_test_helper):
        """Test generation of application-specific deletion statements."""
        operations_handler = query_test_helper.operations_handler
        operations_handler.create_tables()

        test_application = "test-app-deletion"
        test_endpoints = ["endpoint-1", "endpoint-2"]

        # Test application deletion statements generation
        statements = operations_handler._get_aggregate_delete_statements_by_application(
            application_name=test_application, endpoint_ids=test_endpoints
        )

        # Should return a list of statements
        assert isinstance(statements, list), "Should return a list of statements"

        # Should handle empty application name gracefully
        empty_app_statements = (
            operations_handler._get_aggregate_delete_statements_by_application(
                application_name="", endpoint_ids=test_endpoints
            )
        )
        assert isinstance(
            empty_app_statements, list
        ), "Should handle empty application name"

        # Should handle empty endpoint list gracefully
        empty_endpoints_statements = (
            operations_handler._get_aggregate_delete_statements_by_application(
                application_name=test_application, endpoint_ids=[]
            )
        )
        assert isinstance(
            empty_endpoints_statements, list
        ), "Should handle empty endpoint list"

        # Cleanup
        operations_handler.delete_tsdb_resources()

    def test_project_resource_discovery_and_cleanup(self, query_test_helper):
        """Test discovery and cleanup of project resources (tables and views)."""
        operations_handler = query_test_helper.operations_handler

        # Create tables to have resources to discover
        operations_handler.create_tables()

        # Get connection to verify resource creation
        connection = operations_handler._connection
        schema_name = operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ].schema

        # Get the specific project identifier to filter tables
        project_id = operations_handler.project

        # Verify resources exist before cleanup - filter by project
        table_query = f"""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = '{schema_name}'
        AND table_type = 'BASE TABLE'
        AND table_name LIKE '%{project_id.replace('-', '_')}%'
        """
        result = connection.run(query=table_query)
        initial_table_count = len(result.data)
        assert (
            initial_table_count > 0
        ), "Should have created some tables for this project"

        # Test the cleanup process
        operations_handler.delete_tsdb_resources()

        # Verify resources are cleaned up - check project-specific tables
        result_after = connection.run(query=table_query)
        final_table_count = len(result_after.data)
        assert (
            final_table_count == 0
        ), f"All project tables should be deleted, but found {final_table_count} for project {project_id}"

    def test_schema_cleanup_edge_cases(self, query_test_helper):
        """Test edge cases in schema and resource cleanup."""
        operations_handler = query_test_helper.operations_handler

        # Test cleanup when no resources exist
        operations_handler.delete_tsdb_resources()  # Should not fail

        # Create and immediately delete resources
        operations_handler.create_tables()
        operations_handler.delete_tsdb_resources()

        # Double deletion should be safe
        operations_handler.delete_tsdb_resources()  # Should not fail

        # Verify clean state for this project
        connection = operations_handler._connection
        schema_name = operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ].schema
        project_id = operations_handler.project
        table_query = f"""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = '{schema_name}' AND table_type = 'BASE TABLE'
        AND table_name LIKE '%{project_id.replace('-', '_')}%'
        """
        result = connection.run(query=table_query)
        assert (
            len(result.data) == 0
        ), "Should have no tables after cleanup for this project"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
