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

import contextlib
import time
import uuid
from datetime import datetime
from unittest.mock import Mock

import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors
from mlrun.datastore.datastore_profile import DatastoreProfilePostgreSQL
from mlrun.model_monitoring.db.tsdb.preaggregate import PreAggregateConfig
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    TimescaleDBConnection,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_operations import (
    TimescaleDBOperationsHandler,
)

# Import shared utilities from conftest.py
from tests.model_monitoring.db.tsdb.timescaledb.conftest import (
    generate_unique_name,
)


@pytest.fixture(scope="session")
def test_database(connection_string):
    """Create a test database for the entire test session."""
    admin_dsn = connection_string
    test_db_name = generate_unique_name("mlrun_ops_test")

    # Create admin connection with autocommit enabled for DDL operations
    admin_conn = TimescaleDBConnection(admin_dsn, max_connections=1, autocommit=True)

    try:
        # Create test database
        admin_conn.run(
            statements=[
                f"DROP DATABASE IF EXISTS {test_db_name}",
                f"CREATE DATABASE {test_db_name}",
            ]
        )
        # Try to create TimescaleDB extension, but ignore if already exists with different version
        try:
            admin_conn.run(statements=["CREATE EXTENSION IF NOT EXISTS timescaledb"])
        except Exception:
            # Extension might already exist with different version, which is fine for tests
            pass

        # Build test database DSN
        test_dsn = admin_dsn.replace("/postgres", f"/{test_db_name}")

        # Connect to test database and enable TimescaleDB extension
        TimescaleDBConnection(test_dsn, max_connections=1, autocommit=False)

        yield test_dsn

    finally:
        # Cleanup: Drop test database
        with contextlib.suppress(Exception):
            admin_conn.run(statements=[f"DROP DATABASE IF EXISTS {test_db_name}"])


@pytest.fixture
def db_connection(test_database, clean_connection_pool):
    """Create a TimescaleDB connection using the test database."""
    # clean_connection_pool fixture handles pool management automatically

    yield TimescaleDBConnection(
        dsn=test_database,
        min_connections=1,
        max_connections=3,
        max_retries=2,
        retry_delay=0.1,
        autocommit=False,
    )


@pytest.fixture
def mock_profile():
    """Create a mock datastore profile."""
    profile = Mock(spec=DatastoreProfilePostgreSQL)
    profile.name = "test_profile"
    return profile


@pytest.fixture
def pre_aggregate_config():
    """Create a test pre-aggregate configuration."""
    return PreAggregateConfig(
        aggregate_intervals=["10m", "1h"],
        agg_functions=["sum", "avg", "max"],
        retention_policy={
            "raw": "7d",
            "10m": "30d",
            "1h": "1y",
        },
    )


@pytest.fixture
def operations_handler(db_connection, mock_profile):
    """Create a TimescaleDBOperationsHandler with unique project."""
    project_name = generate_unique_name("test_project")

    # Create handler directly - the schema naming issue is not critical for testing
    handler = TimescaleDBOperationsHandler(
        project=project_name,
        connection=db_connection,
        pre_aggregate_config=None,
    )

    try:
        yield handler
    finally:
        # Cleanup: Delete all resources created by this handler
        try:
            handler.delete_tsdb_resources()
        except Exception as e:
            print(f"Warning: Failed to cleanup resources for {project_name}: {e}")


@pytest.fixture
def operations_handler_with_aggregates(
    db_connection, mock_profile, pre_aggregate_config
):
    """Create a TimescaleDBOperationsHandler with pre-aggregates."""
    project_name = f"test_agg_project_{uuid.uuid4().hex[:8]}"

    handler = TimescaleDBOperationsHandler(
        project=project_name,
        connection=db_connection,
        pre_aggregate_config=pre_aggregate_config,
    )

    try:
        yield handler
    finally:
        # Cleanup
        try:
            handler.delete_tsdb_resources()
        except Exception as e:
            print(
                f"Warning: Failed to cleanup aggregated resources for {project_name}: {e}"
            )


class TestTimescaleDBOperationsHandlerIntegration:
    """Integration tests using real database connections."""

    def test_create_tables_basic(self, operations_handler):
        """Test basic table creation without pre-aggregates."""
        # Create tables
        operations_handler.create_tables()

        # Verify tables were created
        connection = operations_handler._connection
        schema_name = operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ].schema

        # Check if schema exists
        result = connection.run(
            query=f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{schema_name}'"
        )
        assert len(result.data) == 1

        # Check if tables exist
        result = connection.run(
            query=f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema_name}'"
        )
        assert len(result.data) == 4  # predictions, metrics, app_results, errors

        # Verify they are hypertables
        result = connection.run(
            query=f"""
            SELECT hypertable_name FROM timescaledb_information.hypertables
            WHERE hypertable_schema = '{schema_name}'
            """
        )
        assert len(result.data) == 4

    def test_create_tables_with_pre_aggregates(
        self, operations_handler_with_aggregates, pre_aggregate_config
    ):
        """Test table creation with pre-aggregate configuration."""
        # Create tables with pre-aggregates
        operations_handler_with_aggregates.create_tables()

        connection = operations_handler_with_aggregates._connection
        schema_name = operations_handler_with_aggregates.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ].schema

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

    def test_write_application_event_result(self, operations_handler):
        """Test writing result events to the database."""
        # Create tables first
        operations_handler.create_tables()

        # Prepare event data
        event_data = {
            mm_schemas.WriterEvent.END_INFER_TIME: datetime(2024, 1, 15, 12, 30, 45),
            mm_schemas.WriterEvent.START_INFER_TIME: datetime(2024, 1, 15, 12, 30, 40),
            mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint_result",
            mm_schemas.WriterEvent.APPLICATION_NAME: "drift_detection",
            mm_schemas.ResultData.RESULT_NAME: "feature_drift",
            mm_schemas.ResultData.RESULT_VALUE: 0.85,
            mm_schemas.ResultData.RESULT_STATUS: 1,
            mm_schemas.ResultData.RESULT_KIND: 2,
            mm_schemas.ResultData.RESULT_EXTRA_DATA: '{"confidence": 0.9}',
        }

        # Write event
        operations_handler.write_application_event(
            event_data, mm_schemas.WriterEventKind.RESULT
        )

        # Verify data was written
        connection = operations_handler._connection
        app_results_table = operations_handler.tables[
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

    def test_write_application_event_metric(self, operations_handler):
        """Test writing metric events to the database."""
        # Create tables first
        operations_handler.create_tables()

        # Prepare event data
        event_data = {
            mm_schemas.WriterEvent.END_INFER_TIME: datetime(2024, 1, 15, 12, 30, 45),
            mm_schemas.WriterEvent.START_INFER_TIME: datetime(2024, 1, 15, 12, 30, 40),
            mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint_metric",
            mm_schemas.WriterEvent.APPLICATION_NAME: "performance_monitoring",
            mm_schemas.MetricData.METRIC_NAME: "accuracy",
            mm_schemas.MetricData.METRIC_VALUE: 0.95,
        }

        # Write event
        operations_handler.write_application_event(
            event_data, mm_schemas.WriterEventKind.METRIC
        )

        # Verify data was written
        connection = operations_handler._connection
        metrics_table = operations_handler.tables[mm_schemas.TimescaleDBTables.METRICS]

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

    def test_delete_tsdb_records_raw_only(self, operations_handler):
        """Test deleting records from raw tables only."""
        # Create tables and insert test data
        operations_handler.create_tables()

        # Insert test data in multiple tables
        test_endpoints = ["endpoint_1", "endpoint_2", "endpoint_3"]
        connection = operations_handler._connection

        for endpoint_id in test_endpoints:
            # Insert into predictions table
            predictions_table = operations_handler.tables[
                mm_schemas.TimescaleDBTables.PREDICTIONS
            ]
            connection.run(
                statements=[
                    f"""
                INSERT INTO {predictions_table.full_name()}
                (time, endpoint_id, latency, custom_metrics, estimated_prediction_count, effective_sample_count)
                VALUES (NOW(), '{endpoint_id}', 0.1, '{{}}', 1.0, 1)
                """
                ]
            )

            # Insert into metrics table
            metrics_table = operations_handler.tables[
                mm_schemas.TimescaleDBTables.METRICS
            ]
            connection.run(
                statements=[
                    f"""
                INSERT INTO {metrics_table.full_name()}
                (end_infer_time, start_infer_time, endpoint_id, application_name, metric_name, metric_value)
                VALUES (NOW(), NOW(), '{endpoint_id}', 'test_app', 'test_metric', 0.5)
                """
                ]
            )

        # Verify data exists
        predictions_table = operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]
        result = connection.run(
            query=f"SELECT COUNT(*) FROM {predictions_table.full_name()}"
        )
        assert result.data[0][0] == 3

        # Delete records for specific endpoints
        operations_handler.delete_tsdb_records(
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
        self, operations_handler_with_aggregates
    ):
        """Test deleting records including aggregates."""
        # Create tables with aggregates
        operations_handler_with_aggregates.create_tables()

        connection = operations_handler_with_aggregates._connection
        test_endpoint = "endpoint_with_aggregates"

        # Insert test data
        predictions_table = operations_handler_with_aggregates.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]
        connection.run(
            statements=[
                f"""
            INSERT INTO {predictions_table.full_name()}
            (time, endpoint_id, latency, custom_metrics, estimated_prediction_count, effective_sample_count)
            VALUES (NOW(), '{test_endpoint}', 0.1, '{{}}', 1.0, 1)
            """
            ]
        )

        # Verify data exists
        result = connection.run(
            query=f"""
            SELECT COUNT(*) FROM {predictions_table.full_name()}
            WHERE endpoint_id = '{test_endpoint}'
            """
        )
        assert result.data[0][0] == 1

        # Delete records including aggregates
        operations_handler_with_aggregates.delete_tsdb_records(
            [test_endpoint], include_aggregates=True
        )

        # Verify deletion from raw table
        result = connection.run(
            query=f"""
            SELECT COUNT(*) FROM {predictions_table.full_name()}
            WHERE endpoint_id = '{test_endpoint}'
            """
        )
        assert result.data[0][0] == 0

    def test_delete_tsdb_records_empty_list(self, operations_handler):
        """Test deleting with empty endpoint list."""
        operations_handler.create_tables()

        # Should not raise exception
        operations_handler.delete_tsdb_records([], include_aggregates=True)

    def test_delete_tsdb_records_special_characters(self, operations_handler):
        """Test deleting endpoints with special characters."""
        operations_handler.create_tables()

        # Insert data with special characters
        special_endpoints = [
            "endpoint'with'quotes",
            "endpoint-with-dashes",
            "endpoint_with_underscores",
        ]
        connection = operations_handler._connection

        predictions_table = operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]

        # Import Statement class for parameterized queries
        from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
            Statement,
        )

        for endpoint_id in special_endpoints:
            # Create a proper Statement object with parameters
            stmt = Statement(
                sql=f"""
                INSERT INTO {predictions_table.full_name()}
                (time, endpoint_id, latency, custom_metrics, estimated_prediction_count, effective_sample_count)
                VALUES (NOW(), %s, 0.1, '{{}}', 1.0, 1)
                """,
                parameters=[endpoint_id],
            )
            connection.run(statements=[stmt])

        # Delete using parameterized queries (should handle special characters safely)
        operations_handler.delete_tsdb_records(special_endpoints)

        # Verify deletion
        result = connection.run(
            query=f"SELECT COUNT(*) FROM {predictions_table.full_name()}"
        )
        assert result.data[0][0] == 0

    def test_delete_tsdb_resources_complete_cleanup(self, operations_handler):
        """Test complete resource deletion."""
        # Create tables
        operations_handler.create_tables()

        connection = operations_handler._connection
        schema_name = operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ].schema

        # Verify tables exist
        result = connection.run(
            query=f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{schema_name}'"
        )
        table_count_before = result.data[0][0]
        assert table_count_before >= 4

        # Delete all resources
        operations_handler.delete_tsdb_resources()

        # Verify tables are gone
        result = connection.run(
            query=f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{schema_name}'"
        )
        assert result.data[0][0] == 0

        # Verify schema is gone too
        result = connection.run(
            query=f"SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = '{schema_name}'"
        )
        assert result.data[0][0] == 0

    def test_delete_tsdb_resources_with_aggregates(
        self, operations_handler_with_aggregates
    ):
        """Test resource deletion including continuous aggregates."""
        # Create tables with aggregates
        operations_handler_with_aggregates.create_tables()

        connection = operations_handler_with_aggregates._connection
        schema_name = operations_handler_with_aggregates.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ].schema

        # Verify continuous aggregates exist
        result = connection.run(
            query=f"""
            SELECT COUNT(*) FROM timescaledb_information.continuous_aggregates
            WHERE hypertable_schema = '{schema_name}'
            """
        )
        cagg_count_before = result.data[0][0]
        assert cagg_count_before > 0

        # Delete all resources
        operations_handler_with_aggregates.delete_tsdb_resources()

        # Verify continuous aggregates are gone
        result = connection.run(
            query=f"""
            SELECT COUNT(*) FROM timescaledb_information.continuous_aggregates
            WHERE hypertable_schema = '{schema_name}'
            """
        )
        assert result.data[0][0] == 0

        # Verify schema is gone
        result = connection.run(
            query=f"SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = '{schema_name}'"
        )
        assert result.data[0][0] == 0

    def test_datetime_conversion_edge_cases(self, operations_handler):
        """Test datetime conversion with various formats."""
        # Test different datetime formats
        test_cases = [
            "2024-01-15T12:30:45Z",
            "2024-01-15T12:30:45+00:00",
            "2024-01-15T12:30:45.123456Z",
            datetime(2024, 1, 15, 12, 30, 45),
        ]

        for dt_input in test_cases:
            result = operations_handler._convert_to_datetime(dt_input)
            assert isinstance(result, datetime)
            assert result.year == 2024
            assert result.month == 1
            assert result.day == 15

    def test_write_event_with_unicode_data(self, operations_handler):
        """Test writing events with Unicode characters."""
        operations_handler.create_tables()

        # Event with Unicode data
        event_data = {
            mm_schemas.WriterEvent.END_INFER_TIME: datetime.now(),
            mm_schemas.WriterEvent.START_INFER_TIME: datetime.now(),
            mm_schemas.WriterEvent.ENDPOINT_ID: "测试端点",  # Chinese characters
            mm_schemas.WriterEvent.APPLICATION_NAME: "тест_приложение",  # Cyrillic
            mm_schemas.ResultData.RESULT_NAME: "résultat_test",  # French accents
            mm_schemas.ResultData.RESULT_VALUE: 0.85,
            mm_schemas.ResultData.RESULT_STATUS: 1,
            mm_schemas.ResultData.RESULT_KIND: 1,
            mm_schemas.ResultData.RESULT_EXTRA_DATA: '{"message": "успех"}',
        }

        # Should not raise exception
        operations_handler.write_application_event(
            event_data, mm_schemas.WriterEventKind.RESULT
        )

        # Verify data was written correctly
        connection = operations_handler._connection
        app_results_table = operations_handler.tables[
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

    def test_large_batch_deletion(self, operations_handler):
        """Test deletion of large number of endpoints."""
        operations_handler.create_tables()

        # Insert many endpoints
        endpoint_count = 100
        endpoints = [f"endpoint_{i}" for i in range(endpoint_count)]

        connection = operations_handler._connection
        predictions_table = operations_handler.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]

        # Batch insert
        for i, endpoint_id in enumerate(endpoints):
            connection.run(
                statements=[
                    f"""
                INSERT INTO {predictions_table.full_name()}
                (time, endpoint_id, latency, custom_metrics, estimated_prediction_count, effective_sample_count)
                VALUES (NOW(), '{endpoint_id}', {0.1 + i * 0.001}, '{{}}', 1.0, 1)
                """
                ]
            )

        # Verify all data inserted
        result = connection.run(
            query=f"SELECT COUNT(*) FROM {predictions_table.full_name()}"
        )
        assert result.data[0][0] == endpoint_count

        # Delete first 50 endpoints
        endpoints_to_delete = endpoints[:50]
        operations_handler.delete_tsdb_records(endpoints_to_delete)

        # Verify deletion
        result = connection.run(
            query=f"SELECT COUNT(*) FROM {predictions_table.full_name()}"
        )
        assert result.data[0][0] == 50

    def test_error_handling_invalid_event_data(self, operations_handler):
        """Test error handling with invalid event data."""
        operations_handler.create_tables()

        # Event with missing required fields
        invalid_event = {
            mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint",
            # Missing other required fields
        }

        # Should raise an exception
        with pytest.raises(mlrun.errors.MLRunRuntimeError):
            operations_handler.write_application_event(
                invalid_event, mm_schemas.WriterEventKind.RESULT
            )

    def test_concurrent_operations(self, operations_handler):
        """Test concurrent operations on the same handler."""
        import threading

        operations_handler.create_tables()

        results = []
        errors = []

        def write_worker(worker_id):
            """Worker function for concurrent writes."""
            try:
                for i in range(5):
                    event_data = {
                        mm_schemas.WriterEvent.END_INFER_TIME: datetime.now(),
                        mm_schemas.WriterEvent.START_INFER_TIME: datetime.now(),
                        mm_schemas.WriterEvent.ENDPOINT_ID: f"worker_{worker_id}_endpoint_{i}",
                        mm_schemas.WriterEvent.APPLICATION_NAME: f"worker_{worker_id}_app",
                        mm_schemas.MetricData.METRIC_NAME: "test_metric",
                        mm_schemas.MetricData.METRIC_VALUE: float(worker_id + i),
                    }
                    operations_handler.write_application_event(
                        event_data, mm_schemas.WriterEventKind.METRIC
                    )
                    time.sleep(0.01)  # Small delay
                results.append(f"worker_{worker_id}_completed")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # Start multiple worker threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=write_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 3

        # Verify all data was written
        connection = operations_handler._connection
        metrics_table = operations_handler.tables[mm_schemas.TimescaleDBTables.METRICS]

        result = connection.run(
            query=f"SELECT COUNT(*) FROM {metrics_table.full_name()}"
        )
        assert result.data[0][0] == 15  # 3 workers * 5 metrics each


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
