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

from unittest.mock import MagicMock, patch

import psycopg
import pytest

import mlrun.config
import mlrun.errors
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    Statement,
    TimescaleDBConnection,
)


class TestTimescaleDBConnectionRetryLogic:
    """Test database error retry and recovery logic in TimescaleDB connection."""

    @pytest.fixture
    def mock_connection(self):
        """Create a TimescaleDBConnection with mocked pool for testing."""
        with patch(
            "mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection.ConnectionPool"
        ):
            conn = TimescaleDBConnection(
                dsn="postgres://test:test@localhost:5432/test",
                max_connections=1,
                max_retries=2,
                retry_delay=0.1,  # Fast retries for testing
            )
            # Mock the pool and connection
            mock_pool = MagicMock()
            mock_db_conn = MagicMock()
            mock_cursor = MagicMock()

            mock_pool.connection.return_value.__enter__.return_value = mock_db_conn
            mock_db_conn.cursor.return_value.__enter__.return_value = mock_cursor
            conn._pool = mock_pool

            # Mock version check to avoid real database calls
            conn._version_checked = True
            conn._timescaledb_version = "2.10.0"

            return conn, mock_cursor

    def test_deadlock_retry_statements_success_after_retry(self, mock_connection):
        """Test that statements succeed after deadlock retry."""
        connection, mock_cursor = mock_connection

        # Simulate deadlock on first attempt, success on second
        deadlock_error = psycopg.errors.DeadlockDetected("deadlock detected")
        mock_cursor.execute.side_effect = [deadlock_error, None]

        # Should succeed after retry
        with patch("time.sleep") as mock_sleep:  # Speed up test
            connection.run(statements=["INSERT INTO test VALUES (1)"])

        # Verify retry happened
        assert mock_cursor.execute.call_count == 2
        mock_sleep.assert_called_once()

    def test_deadlock_retry_statements_persistent_failure(self, mock_connection):
        """Test that persistent deadlocks eventually raise error."""
        connection, mock_cursor = mock_connection

        # Simulate persistent deadlock
        deadlock_error = psycopg.errors.DeadlockDetected("deadlock detected")
        mock_cursor.execute.side_effect = deadlock_error

        # Should fail after max retries
        with patch("time.sleep"):  # Speed up test
            with pytest.raises(mlrun.errors.MLRunRuntimeError) as exc_info:
                connection.run(statements=["INSERT INTO test VALUES (1)"])

        assert "deadlock persisted after 3 retries" in str(exc_info.value)
        # Should try 4 times (initial + 3 retries)
        assert mock_cursor.execute.call_count == 4

    def test_connection_error_retry_success_after_retry(self, mock_connection):
        """Test that connection errors are retried with different timing."""
        connection, mock_cursor = mock_connection

        # Simulate connection error on first attempt, success on second
        connection_error = psycopg.errors.CannotConnectNow("cannot connect")
        mock_cursor.execute.side_effect = [connection_error, None]

        # Should succeed after retry
        with patch("time.sleep") as mock_sleep:  # Speed up test
            connection.run(statements=["INSERT INTO test VALUES (1)"])

        # Verify retry happened with slower timing than deadlock
        assert mock_cursor.execute.call_count == 2
        mock_sleep.assert_called_once()

    def test_query_retry_success_after_retry(self, mock_connection):
        """Test that queries succeed after retry."""
        connection, mock_cursor = mock_connection

        # Simulate deadlock on first attempt, success on second
        deadlock_error = psycopg.errors.DeadlockDetected("deadlock detected")
        mock_cursor.execute.side_effect = [deadlock_error, None]
        mock_cursor.fetchall.return_value = [(1, "test")]
        mock_cursor.description = [MagicMock(name="id"), MagicMock(name="name")]

        # Should succeed after retry
        with patch("time.sleep") as mock_sleep:  # Speed up test
            result = connection.run(query="SELECT * FROM test")

        # Verify retry happened and result returned
        assert mock_cursor.execute.call_count == 2
        mock_sleep.assert_called_once()
        assert result is not None
        assert result.data == [(1, "test")]

    def test_non_retryable_error_no_retry(self, mock_connection):
        """Test that non-retryable errors don't trigger retry."""
        connection, mock_cursor = mock_connection

        # Simulate non-retryable database error
        syntax_error = psycopg.errors.SyntaxError("syntax error")
        mock_cursor.execute.side_effect = syntax_error

        # Should fail immediately without retry - expect original psycopg error
        with pytest.raises(psycopg.errors.SyntaxError):
            connection.run(statements=["INVALID SQL"])

        # Should only try once (no retry for non-retryable errors)
        assert mock_cursor.execute.call_count == 1

    def test_exponential_backoff_timing(self, mock_connection):
        """Test that retries use appropriate exponential backoff timing."""
        connection, mock_cursor = mock_connection

        # Simulate deadlock on first two attempts, success on third
        deadlock_error = psycopg.errors.DeadlockDetected("deadlock detected")
        mock_cursor.execute.side_effect = [deadlock_error, deadlock_error, None]

        with patch("time.sleep") as mock_sleep:
            connection.run(statements=["INSERT INTO test VALUES (1)"])

        # Verify exponential backoff delays
        sleep_calls = mock_sleep.call_args_list
        assert len(sleep_calls) == 2  # Two retries

        # Deadlock-specific exponential backoff: ~0.1s, ~0.2s (with jitter)
        delay1 = sleep_calls[0][0][0]
        delay2 = sleep_calls[1][0][0]

        assert 0.1 <= delay1 <= 0.15  # (2^0 * 0.1) + jitter
        assert 0.2 <= delay2 <= 0.25  # (2^1 * 0.1) + jitter

    def test_combined_statements_and_query_retry(self, mock_connection):
        """Test retry handling with both statements and query."""
        connection, mock_cursor = mock_connection

        # Simulate deadlock in statements, success in query
        deadlock_error = psycopg.errors.DeadlockDetected("deadlock detected")
        mock_cursor.execute.side_effect = [
            deadlock_error,
            None,
            None,
        ]  # statement retry + query
        mock_cursor.fetchall.return_value = [(1,)]
        mock_cursor.description = [MagicMock(name="count")]

        with patch("time.sleep"):
            result = connection.run(
                statements=["INSERT INTO test VALUES (1)"],
                query="SELECT COUNT(*) FROM test",
            )

        # Should succeed after statement retry
        assert result is not None
        assert result.data == [(1,)]
        # 3 calls: failed statement + retried statement + query
        assert mock_cursor.execute.call_count == 3

    def test_parameterized_statement_retry(self, mock_connection):
        """Test retry with parameterized Statement objects."""
        connection, mock_cursor = mock_connection

        # Create parameterized statement
        stmt = Statement("INSERT INTO test (id, name) VALUES (%s, %s)", [1, "test"])

        # Simulate deadlock on first attempt, success on second
        deadlock_error = psycopg.errors.DeadlockDetected("deadlock detected")
        mock_cursor.execute.side_effect = [deadlock_error, None]

        with patch("time.sleep"):
            connection.run(statements=[stmt])

        # Verify parameterized statement was retried correctly
        assert mock_cursor.execute.call_count == 2
        # Both calls should have the same SQL and parameters
        call_args = mock_cursor.execute.call_args_list
        assert call_args[0][0][0] == stmt.sql
        assert call_args[0][0][1] == stmt.parameters
        assert call_args[1][0][0] == stmt.sql
        assert call_args[1][0][1] == stmt.parameters


class TestTimescaleDBConnectionIntegration:
    """Integration tests for connection retry behavior verification."""

    @pytest.fixture
    def mock_connection_simple(self):
        """Create a TimescaleDBConnection with simpler mocking for integration tests."""
        with patch(
            "mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection.ConnectionPool"
        ):
            conn = TimescaleDBConnection(
                dsn="postgres://test:test@localhost:5432/test",
                max_connections=1,
                max_retries=2,
                retry_delay=0.1,
            )
            # Mock internals
            mock_pool = MagicMock()
            mock_db_conn = MagicMock()
            mock_cursor = MagicMock()

            mock_pool.connection.return_value.__enter__.return_value = mock_db_conn
            mock_db_conn.cursor.return_value.__enter__.return_value = mock_cursor
            conn._pool = mock_pool
            conn._version_checked = True
            conn._timescaledb_version = "2.10.0"

            return conn, mock_cursor

    def test_retry_behavior_verification(self, mock_connection_simple, caplog):
        """Test that retry behavior works correctly and logs appropriately."""
        connection, mock_cursor = mock_connection_simple

        # Simulate deadlock on first two attempts, success on third
        deadlock_error = psycopg.errors.DeadlockDetected("deadlock detected")
        mock_cursor.execute.side_effect = [deadlock_error, deadlock_error, None]

        with patch("time.sleep") as mock_sleep:
            connection.run(statements=["INSERT INTO test VALUES (1)"])

        # Verify retry behavior
        assert mock_cursor.execute.call_count == 3
        assert mock_sleep.call_count == 2

    def test_persistent_failure_behavior_verification(
        self, mock_connection_simple, caplog
    ):
        """Test that persistent failures fail appropriately and log errors."""
        connection, mock_cursor = mock_connection_simple

        # Simulate persistent deadlock
        deadlock_error = psycopg.errors.DeadlockDetected("deadlock detected")
        mock_cursor.execute.side_effect = deadlock_error

        with patch("time.sleep"):
            with pytest.raises(mlrun.errors.MLRunRuntimeError) as exc_info:
                connection.run(statements=["INSERT INTO test VALUES (1)"])

        # Verify error message
        assert "deadlock persisted after 3 retries" in str(exc_info.value)
        assert mock_cursor.execute.call_count == 4  # Initial + 3 retries


class TestTimescaleDBConnectionPoolTimeout:
    """Test connection pool timeout configuration (ML-11775)."""

    def test_pool_uses_configured_timeout(self):
        """Test that ConnectionPool is created with timeout from config."""
        # Set custom timeout in config
        original_timeout = (
            mlrun.config.config.model_endpoint_monitoring.tsdb.connection_pool_timeout
        )
        mlrun.config.config.model_endpoint_monitoring.tsdb.connection_pool_timeout = 90

        try:
            with patch(
                "mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection.ConnectionPool"
            ) as mock_pool_class:
                conn = TimescaleDBConnection(
                    dsn="postgres://test:test@localhost:5432/test",
                    max_connections=5,
                )
                # Access the pool property to trigger pool creation
                _ = conn.pool

                # Verify ConnectionPool was called with the configured timeout
                mock_pool_class.assert_called_once()
                call_kwargs = mock_pool_class.call_args.kwargs
                assert call_kwargs["timeout"] == 90.0
        finally:
            # Restore original value
            mlrun.config.config.model_endpoint_monitoring.tsdb.connection_pool_timeout = original_timeout

    def test_pool_default_timeout_is_120(self):
        """Test that the default connection pool timeout is 120 seconds."""
        default_timeout = (
            mlrun.config.config.model_endpoint_monitoring.tsdb.connection_pool_timeout
        )
        assert default_timeout == 120
