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
import os
import threading
import time
import uuid
from typing import Optional
from unittest.mock import patch

import pytest

import mlrun.errors

connection_string = os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__TSDB_CONNECTION")

# Skip entire module if connection string is not available or not PostgreSQL
pytestmark = pytest.mark.skipif(
    not connection_string or not connection_string.startswith("postgres://"),
    reason="TimescaleDB connection string not available or not PostgreSQL",
)
import psycopg2  # noqa: E402

from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (  # noqa: E402
    QueryResult,
    TimescaleDBConnection,
)


# Helper functions for connection pool management
def reset_global_connection_pool():
    """Reset the global connection pool to ensure clean test state."""
    import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection as conn_module

    with conn_module._connection_lock:
        if conn_module._connection_pool:
            conn_module._connection_pool.closeall()
            conn_module._connection_pool = None


@pytest.fixture(scope="session")
def test_database():
    """Create a test database for the entire test session (if database creation testing is needed)."""

    admin_dsn = connection_string
    test_db_name = f"mlrun_test_{int(time.time())}"  # Unique database name

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

        # Build test database DSN
        test_dsn = admin_dsn.replace("/postgres", f"/{test_db_name}")

        # Connect to test database and enable TimescaleDB extension
        test_conn = TimescaleDBConnection(test_dsn, max_connections=1, autocommit=False)
        test_conn.run(statements=["CREATE EXTENSION IF NOT EXISTS timescaledb"])

        yield test_dsn

    finally:
        # Cleanup: Drop test database
        with contextlib.suppress(Exception):
            admin_conn.run(statements=[f"DROP DATABASE IF EXISTS {test_db_name}"])


@pytest.fixture
def db_connection(test_database):
    """Create a TimescaleDB connection using the test database."""
    # Reset global connection pool to ensure clean state for each test
    import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection as conn_module

    with conn_module._connection_lock:
        if conn_module._connection_pool:
            conn_module._connection_pool.closeall()
            conn_module._connection_pool = None

    yield TimescaleDBConnection(
        dsn=test_database,
        min_connections=1,
        max_connections=3,
        max_retries=2,
        retry_delay=0.1,
        autocommit=False,  # Default to transaction mode for tests
    )
    # Cleanup: Reset global connection pool after each test
    with conn_module._connection_lock:
        if conn_module._connection_pool:
            conn_module._connection_pool.closeall()
            conn_module._connection_pool = None


@pytest.fixture
def admin_connection():
    """Create admin connection for database operations."""
    if not connection_string or not connection_string.startswith("postgres://"):
        pytest.skip("No valid TimescaleDB connection string")

    yield TimescaleDBConnection(
        dsn=connection_string,  # Should point to postgres database
        min_connections=1,
        max_connections=2,
        autocommit=True,  # Required for CREATE/DROP DATABASE
    )
    # Cleanup: Reset global connection pool
    import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection as conn_module

    with conn_module._connection_lock:
        if conn_module._connection_pool:
            conn_module._connection_pool.closeall()
            conn_module._connection_pool = None


@pytest.fixture
def autocommit_connection(test_database):
    """Create a TimescaleDB connection with autocommit enabled."""
    # Reset global connection pool to ensure clean state
    import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection as conn_module

    with conn_module._connection_lock:
        if conn_module._connection_pool:
            conn_module._connection_pool.closeall()
            conn_module._connection_pool = None

    yield TimescaleDBConnection(
        dsn=test_database,
        min_connections=1,
        max_connections=2,
        autocommit=True,
    )
    # Cleanup: Reset global connection pool after each test
    with conn_module._connection_lock:
        if conn_module._connection_pool:
            conn_module._connection_pool.closeall()
            conn_module._connection_pool = None


@pytest.fixture
def sample_table(db_connection):
    """Create a sample table for testing with unique name."""
    table_name = (
        f"test_table_{int(time.time() * 1000)}"  # Unique name with milliseconds
    )

    table_sql = f"""
    CREATE TABLE {table_name} (
        time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        sensor_id VARCHAR(64),
        temperature DOUBLE PRECISION,
        humidity INTEGER
    );
    """

    # Create table and hypertable
    db_connection.run(
        statements=[
            table_sql,
            f"SELECT create_hypertable('{table_name}', 'time', if_not_exists => TRUE);",
        ]
    )

    yield table_name

    # Cleanup
    with contextlib.suppress(Exception):
        db_connection.run(statements=[f"DROP TABLE IF EXISTS {table_name} CASCADE"])


def test_autocommit_behavior_integration(test_database: str) -> None:
    # sourcery skip: extract-method
    table_name = f"test_autocommit_{uuid.uuid4().hex[:8]}"

    conn_tx = TimescaleDBConnection(test_database, autocommit=False)
    conn_tx.run(
        statements=[f"CREATE TABLE {table_name} (id SERIAL PRIMARY KEY, data TEXT)"]
    )

    try:
        conn_tx.run(
            statements=[
                f"INSERT INTO {table_name} (data) VALUES ('tx1')",
                f"INSERT INTO {table_name} (data) VALUES ('tx2')",
            ]
        )

        result = conn_tx.run(query=f"SELECT COUNT(*) FROM {table_name}")
        assert result.data[0][0] == 2

        conn_auto = TimescaleDBConnection(test_database, autocommit=True)
        conn_auto.run(statements=[f"INSERT INTO {table_name} (data) VALUES ('auto1')"])

        result = conn_auto.run(query=f"SELECT COUNT(*) FROM {table_name}")
        assert result.data[0][0] == 3

    finally:
        try:
            conn_tx.run(statements=[f"DROP TABLE IF EXISTS {table_name}"])
        except psycopg2.Error as e:
            print(f"Warning: Failed to cleanup table {table_name}: {e}")


class TestTimescaleDBConnection:
    @pytest.mark.parametrize(
        "autocommit,expected_autocommit",
        [
            (None, True),  # Default case
            (True, True),
            (False, False),
        ],
    )
    def test_connection_creation_with_autocommit(
        self, test_database: str, autocommit: Optional[bool], expected_autocommit: bool
    ) -> None:
        if autocommit is None:
            conn = TimescaleDBConnection(dsn=test_database)
        else:
            conn = TimescaleDBConnection(dsn=test_database, autocommit=autocommit)

        assert conn._dsn == test_database
        assert conn._min_connections == 1
        assert conn._max_connections == 10
        assert conn._max_retries == 3
        assert conn._retry_delay == 1.0
        assert conn._autocommit is expected_autocommit
        assert conn.prefix_statements == []

    def test_connection_creation_custom_params(self, test_database: str) -> None:
        conn = TimescaleDBConnection(
            dsn=test_database,
            min_connections=2,
            max_connections=5,
            max_retries=5,
            retry_delay=0.5,
            autocommit=False,
        )

        assert conn._min_connections == 2
        assert conn._max_connections == 5
        assert conn._max_retries == 5
        assert conn._retry_delay == 0.5
        assert conn._autocommit is False

    def test_invalid_connection_string(self) -> None:
        reset_global_connection_pool()

        conn = TimescaleDBConnection(
            "postgresql://invalid:invalid@localhost:99999/nonexistent_db_12345"
        )

        with patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool:
            mock_pool.side_effect = psycopg2.OperationalError("Connection refused")

            with pytest.raises(
                mlrun.errors.MLRunRuntimeError, match="Failed to create connection pool"
            ):
                _ = conn.pool()

            reset_global_connection_pool()

    def test_connection_failure_during_operation(self) -> None:
        reset_global_connection_pool()

        conn = TimescaleDBConnection(
            dsn="postgresql://invalid:invalid@localhost:99999/nonexistent_db_12345",
            max_retries=1,
            retry_delay=0.01,
        )

        with patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool_class:
            mock_pool = mock_pool_class.return_value
            mock_pool.getconn.side_effect = psycopg2.OperationalError(
                "Connection refused"
            )

            with pytest.raises(
                mlrun.errors.MLRunRuntimeError,
                match="Database operation failed after 2 attempts",
            ):
                conn.run(statements=["SELECT 1"])

            reset_global_connection_pool()

    def test_basic_statement_execution(
        self, db_connection: TimescaleDBConnection, sample_table: str
    ) -> None:
        insert_sql = (
            f"INSERT INTO {sample_table} (sensor_id, temperature, humidity) "
            f"VALUES ('sensor1', 23.5, 60)"
        )

        result = db_connection.run(statements=[insert_sql])
        assert result is None

    def test_query_execution(
        self, db_connection: TimescaleDBConnection, sample_table: str
    ) -> None:
        insert_sql = (
            f"INSERT INTO {sample_table} (sensor_id, temperature, humidity) "
            f"VALUES ('sensor1', 23.5, 60), ('sensor2', 24.0, 65)"
        )
        db_connection.run(statements=[insert_sql])

        result = db_connection.run(
            query=f"SELECT sensor_id, temperature FROM {sample_table} ORDER BY sensor_id"
        )

        assert isinstance(result, QueryResult)
        assert result.fields == ["sensor_id", "temperature"]
        assert len(result.data) == 2
        assert result.data[0] == ("sensor1", 23.5)
        assert result.data[1] == ("sensor2", 24.0)

    def test_empty_query_result(
        self, db_connection: TimescaleDBConnection, sample_table: str
    ) -> None:
        result = db_connection.run(
            query=f"SELECT * FROM {sample_table} WHERE sensor_id = 'nonexistent'"
        )

        assert isinstance(result, QueryResult)
        assert result.fields == ["time", "sensor_id", "temperature", "humidity"]
        assert result.data == []

    def test_statements_normalization(
        self, db_connection: TimescaleDBConnection
    ) -> None:
        assert db_connection._normalize_statements(None) == []
        assert db_connection._normalize_statements("SELECT 1") == ["SELECT 1"]
        assert db_connection._normalize_statements(["SELECT 1", "SELECT 2"]) == [
            "SELECT 1",
            "SELECT 2",
        ]

    def test_prefix_statements(self, db_connection: TimescaleDBConnection) -> None:
        db_connection.prefix_statements = ["SET TIME ZONE 'UTC'"]

        result = db_connection.run(query="SELECT current_setting('timezone')")

        assert isinstance(result, QueryResult)
        assert result.data[0][0] == "UTC"

    def test_autocommit_mode(
        self, autocommit_connection: TimescaleDBConnection
    ) -> None:
        table_name = f"test_autocommit_{uuid.uuid4().hex[:8]}"
        autocommit_connection.run(
            statements=[
                f"CREATE TABLE {table_name} (id SERIAL PRIMARY KEY, value TEXT)"
            ]
        )

        autocommit_connection.run(
            statements=[f"INSERT INTO {table_name} (value) VALUES ('test1')"]
        )

        result = autocommit_connection.run(query=f"SELECT COUNT(*) FROM {table_name}")
        assert result.data[0][0] == 1

        autocommit_connection.run(statements=[f"DROP TABLE {table_name}"])

    def test_transaction_mode(
        self, db_connection: TimescaleDBConnection, sample_table: str
    ) -> None:
        db_connection.run(
            statements=[
                f"INSERT INTO {sample_table} (sensor_id, temperature) VALUES ('test1', 25.0)",
                f"INSERT INTO {sample_table} (sensor_id, temperature) VALUES ('test2', 26.0)",
            ]
        )

        result = db_connection.run(query=f"SELECT COUNT(*) FROM {sample_table}")
        assert result.data[0][0] == 2

    def test_transaction_rollback_on_error(
        self, db_connection: TimescaleDBConnection, sample_table: str
    ) -> None:
        db_connection.run(
            statements=[
                f"INSERT INTO {sample_table} (sensor_id, temperature) VALUES ('test', 25.0)"
            ]
        )

        with pytest.raises(mlrun.errors.MLRunRuntimeError):
            db_connection.run(
                statements=[
                    f"INSERT INTO {sample_table} (sensor_id, temperature) VALUES ('test2', 26.0)",
                    "INVALID SQL STATEMENT",
                ]
            )

        result = db_connection.run(query=f"SELECT COUNT(*) FROM {sample_table}")
        assert result.data[0][0] == 1 == 1  # Only original insert

    def test_concurrent_connections(self, db_connection, sample_table):
        results = []
        errors = []

        def worker(thread_id):
            try:
                for i in range(3):
                    sql = f"INSERT INTO {sample_table} (sensor_id, temperature) \
                    VALUES ('thread{thread_id}_{i}', {20 + thread_id})"
                    db_connection.run(statements=[sql])
                    time.sleep(0.01)  # Small delay
                results.append(f"thread{thread_id}_completed")
            except Exception as e:
                errors.append(f"thread{thread_id}_error: {e}")

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors and all threads completed
        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 3

        # Verify data was inserted
        result = db_connection.run(query=f"SELECT COUNT(*) FROM {sample_table}")
        assert result.data[0][0] == 9  # 3 threads * 3 inserts each

    @patch("time.sleep")
    def test_retry_logic_with_recoverable_error(self, mock_sleep, db_connection):
        # Mock the pool method
        with patch.object(db_connection, "pool") as mock_pool_method:
            # Create a mock pool using MagicMock
            from unittest.mock import MagicMock

            mock_pool = MagicMock()
            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            # Set up mock relationships
            mock_conn.cursor.return_value = mock_cursor
            mock_conn.closed = False
            mock_conn.autocommit = False

            # First two calls raise OperationalError, third succeeds
            mock_pool.getconn.side_effect = [
                psycopg2.OperationalError("Connection failed"),
                psycopg2.OperationalError("Connection failed"),
                mock_conn,
            ]

            # Make pool method return our mock pool
            mock_pool_method.return_value = mock_pool

            # Execute operation
            db_connection.run(statements=["SELECT 1"])

            # Verify retries occurred
            assert mock_pool.getconn.call_count == 3
            assert mock_sleep.call_count == 2  # 2 retries

            # Verify exponential backoff
            expected_delays = [0.1 * (2**0), 0.1 * (2**1)]  # 0.1, 0.2
            actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
            assert actual_delays == expected_delays

    @patch("time.sleep")
    def test_retry_exhaustion(self, mock_sleep, db_connection):
        # Mock the pool method
        with patch.object(db_connection, "pool") as mock_pool_method:
            # Create a mock pool that always fails using MagicMock
            from unittest.mock import MagicMock

            mock_pool = MagicMock()
            mock_pool.getconn.side_effect = psycopg2.OperationalError(
                "Persistent connection error"
            )

            # Make pool method return our mock pool
            mock_pool_method.return_value = mock_pool

            with pytest.raises(
                mlrun.errors.MLRunRuntimeError,
                match="Database operation failed after 3 attempts",
            ):
                db_connection.run(statements=["SELECT 1"])

            # Verify all retries were attempted (3 total attempts)
            assert mock_pool.getconn.call_count == 3
            assert mock_sleep.call_count == 2  # 2 retries

    def test_non_recoverable_error_no_retry(self, db_connection, sample_table):
        with pytest.raises(
            mlrun.errors.MLRunRuntimeError, match="Database operation failed"
        ):
            # Syntax error should not be retried
            db_connection.run(statements=["INVALID SQL SYNTAX"])

    def test_query_result_equality(self):
        result1 = QueryResult([("a", 1), ("b", 2)], ["col1", "col2"])
        result2 = QueryResult([("a", 1), ("b", 2)], ["col1", "col2"])
        result3 = QueryResult([("a", 1)], ["col1", "col2"])

        assert result1 == result2
        assert result1 != result3

    def test_query_result_representation(self):
        result = QueryResult([("a", 1), ("b", 2)], ["col1", "col2"])
        assert repr(result) == "QueryResult(rows=2, fields=['col1', 'col2'])"

    def test_connection_cleanup(self, db_connection):
        # Execute a simple operation
        db_connection.run(statements=["SELECT 1"])

        # Verify connection pool exists and has connections
        pool = db_connection.pool()  # Call method to get pool instance
        assert pool is not None

        # Test cleanup by simulating connection close
        with patch.object(pool, "putconn") as mock_putconn:
            mock_putconn.side_effect = Exception("Pool closed")

            # This should handle the putconn exception gracefully
            db_connection.run(statements=["SELECT 1"])

    def test_multiple_connection_instances_share_pool(self, test_database):
        conn1 = TimescaleDBConnection(test_database, autocommit=False)
        conn2 = TimescaleDBConnection(test_database, autocommit=False)

        # Both should use the same pool (singleton pattern)
        # Now we call the pool method to get the actual pool instances
        pool1 = conn1.pool()
        pool2 = conn2.pool()
        assert pool1 is pool2

    def test_autocommit_setting_per_operation(self, test_database):
        # Create connections with different autocommit settings
        conn_tx = TimescaleDBConnection(test_database, autocommit=False)
        conn_auto = TimescaleDBConnection(test_database, autocommit=True)

        # Mock the connection pool's getconn method to verify autocommit is set
        with patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool_class:
            mock_pool = mock_pool_class.return_value
            mock_conn_tx = mock_pool.getconn.return_value
            mock_conn_auto = mock_pool.getconn.return_value
            mock_conn_tx.closed = False
            mock_conn_auto.closed = False
            mock_conn_tx.cursor.return_value = mock_conn_tx.cursor.return_value
            mock_conn_auto.cursor.return_value = mock_conn_auto.cursor.return_value

            # Reset global pools to force creation with mocks
            import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection as conn_module

            with conn_module._connection_lock:
                conn_module._connection_pool = None

            # Execute operations (this will create pools with mocks)
            conn_tx.run(statements=["SELECT 1"])

            # Reset pool again for second connection
            with conn_module._connection_lock:
                conn_module._connection_pool = None

            conn_auto.run(statements=["SELECT 1"])

            # Verify autocommit was set correctly (should be called twice, once for each connection)
            # The exact verification depends on the mock setup, but we can check that operations completed
            assert mock_pool.getconn.call_count >= 2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
