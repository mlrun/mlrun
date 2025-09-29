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

import random
import time
from typing import Any, Callable, Optional, Union

import pandas as pd
import psycopg
import semver
from psycopg_pool import ConnectionPool

import mlrun.errors
from mlrun.model_monitoring.db.tsdb.preaggregate import PreAggregateManager
from mlrun.utils import logger


class QueryResult:
    """Container for query results with field metadata."""

    def __init__(self, data: list[tuple], fields: list[str]):
        self.data = data
        self.fields = fields

    def __eq__(self, other):
        return self.data == other.data and self.fields == other.fields

    def __repr__(self):
        return f"QueryResult(rows={len(self.data)}, fields={self.fields})"


class Statement:
    """
    Represents a parameterized statement for TimescaleDB.

    This class encapsulates SQL statements with parameters, providing a clean
    interface
    """

    def __init__(
        self,
        sql: str,
        parameters: Optional[Union[tuple, list, dict]] = None,
        execute_many: bool = False,
    ):
        """
        Initialize a parameterized statement.

        :param sql: SQL query with parameter placeholders. Use %(name)s for named parameters
                   or %s for positional parameters.
        :param parameters: Parameters for the SQL statement. Can be:
                         - tuple/list for positional parameters
                         - dict for named parameters
                         - list of tuples/dicts for execute_many=True
        :param execute_many: If True, expects parameters to be a sequence of parameter sets
                           for batch execution using executemany()
        """
        self.sql = sql
        self.parameters = parameters
        self.execute_many = execute_many

    def execute(self, cursor) -> None:
        """Execute the statement using the provided cursor."""
        if self.execute_many:
            if not isinstance(self.parameters, (list, tuple)):
                raise ValueError(
                    "execute_many=True requires parameters to be a sequence"
                )
            cursor.executemany(self.sql, self.parameters)
        else:
            cursor.execute(self.sql, self.parameters)


class TimescaleDBConnection:
    """
    TimescaleDB connection with shared connection pool and parameterized query support.

    """

    # TimescaleDB version requirements
    MIN_TIMESCALEDB_VERSION = (
        "2.7.0"  # Minimum version with finalized continuous aggregates
    )

    # Deadlock retry configuration
    MAX_DEADLOCK_RETRIES = 3  # Maximum deadlock-specific retry attempts

    def __init__(
        self,
        dsn: str,
        min_connections: int = 1,
        max_connections: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        autocommit: bool = False,
    ):
        self._dsn = dsn
        self._min_connections = min_connections
        self._max_connections = max_connections
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self.prefix_statements: list[Union[str, Statement]] = []
        self._autocommit = autocommit

        # Connection pools (lazy initialization)
        self._pool: Optional[ConnectionPool] = None
        self._timescaledb_version: Optional[str] = None
        self._version_checked: bool = False

    @property
    def pool(self) -> ConnectionPool:
        """Get or create the synchronous connection pool."""
        if self._pool is None:
            self._pool = ConnectionPool(
                conninfo=self._dsn,
                min_size=self._min_connections,
                max_size=self._max_connections,
                timeout=30.0,
            )
        return self._pool

    def _parse_version(self, version_string: str) -> semver.VersionInfo:
        """Parse TimescaleDB version string using semver."""
        try:
            # Handle versions like "2.22.0", "2.7.1-dev", etc.
            # semver.VersionInfo.parse handles pre-release versions automatically
            return semver.VersionInfo.parse(version_string)
        except ValueError as e:
            raise mlrun.errors.MLRunRuntimeError(
                f"Invalid TimescaleDB version format: {version_string}"
            ) from e

    def _check_timescaledb_version(self) -> None:
        if self._version_checked:
            return

        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    # Check if TimescaleDB extension is installed
                    cursor.execute(
                        "SELECT extversion FROM pg_extension WHERE extname = %s",
                        ("timescaledb",),
                    )
                    result = cursor.fetchone()
        except psycopg.Error as e:
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed to check TimescaleDB version: {e}"
            ) from e

        if not result:
            raise mlrun.errors.MLRunRuntimeError(
                "TimescaleDB extension is not installed"
            )

        self._timescaledb_version = result[0]

        # Version processing logic outside try/catch - not a database operation
        # _timescaledb_version is guaranteed to be non-None at this point
        current_version = self._parse_version(self._timescaledb_version)  # type: ignore[arg-type]
        min_version = self._parse_version(self.MIN_TIMESCALEDB_VERSION)

        if current_version < min_version:
            raise mlrun.errors.MLRunRuntimeError(
                f"TimescaleDB version {self._timescaledb_version} is not supported. "
                f"Minimum required version: {self.MIN_TIMESCALEDB_VERSION} "
                f"(required for finalized continuous aggregates)"
            )

        self._version_checked = True

    @property
    def timescaledb_version(self) -> Optional[str]:
        """Get the TimescaleDB version (triggers version check if not done)."""
        if not self._version_checked:
            self._check_timescaledb_version()
        return self._timescaledb_version

    def run(
        self,
        statements: Optional[Union[str, Statement, list[Union[str, Statement]]]] = None,
        query: Optional[Union[str, Statement]] = None,
    ) -> Optional[QueryResult]:
        """
        Execute statements and optionally return query results with deadlock-aware retry logic.

        Supports both string SQL and parameterized Statement objects.
        Uses deadlock-specific retry logic for optimal performance.

        :param statements: SQL statements to execute. Can be:
                         - str: Simple SQL string
                         - Statement: Parameterized statement
                         - list: Mix of str and Statement objects
        :param query: Optional query to execute after statements. Can be str or Statement.
        :return: QueryResult if query provided, None otherwise
        """
        # Perform version check on first use
        if not self._version_checked:
            self._check_timescaledb_version()

        if statements := self._normalize_statements(statements):
            self._execute_with_retry(
                cursor_operation_callable=lambda cursor: self._execute_statements(
                    cursor, statements
                ),
                operation_name="statements",
            )

        # Execute query with retry logic for recoverable errors
        if query:
            return self._execute_with_retry(
                cursor_operation_callable=lambda cursor: self._execute_query(
                    cursor, query
                ),
                operation_name="query",
            )

        return None

    def _normalize_statements(
        self, statements: Optional[Union[str, Statement, list[Union[str, Statement]]]]
    ) -> list[Union[str, Statement]]:
        """Convert statements to a normalized list format."""
        if statements is None:
            return []
        return [statements] if isinstance(statements, (str, Statement)) else statements

    def _execute_operation(
        self,
        statements: list[Union[str, Statement]],
        query: Optional[Union[str, Statement]],
    ) -> Optional[QueryResult]:
        """Execute a single database operation (statements + optional query)."""
        with self.pool.connection() as conn:
            conn.autocommit = self._autocommit

            with conn.cursor() as cursor:
                self._execute_statements(cursor, statements)
                if not self._autocommit:
                    conn.commit()
                return self._execute_query(cursor, query) if query else None

    def _execute_statements(
        self, cursor, statements: list[Union[str, Statement]]
    ) -> None:
        """Execute prefix statements and main statements."""
        # Execute prefix statements
        for stmt in self.prefix_statements:
            if isinstance(stmt, Statement):
                stmt.execute(cursor)
            else:
                cursor.execute(stmt)

        # Execute main statements
        for statement in statements:
            if isinstance(statement, Statement):
                statement.execute(cursor)
            else:
                cursor.execute(statement)

    def _execute_query(self, cursor, query: Union[str, Statement]) -> QueryResult:
        """Execute a query and return formatted results."""
        if isinstance(query, Statement):
            query.execute(cursor)
        else:
            cursor.execute(query)

        if cursor.description:
            field_names = [desc.name for desc in cursor.description]
            results = cursor.fetchall()
            data = [tuple(row) for row in results]
            return QueryResult(data, field_names)
        else:
            return QueryResult([], [])

    def execute_with_fallback(
        self,
        pre_aggregate_manager: PreAggregateManager,
        pre_agg_query_builder: Callable[[], str],
        raw_query_builder: Callable[[], str],
        interval: Optional[str] = None,
        agg_funcs: Optional[list[str]] = None,
        column_mapping_rules: Optional[dict[str, list[str]]] = None,
        debug_name: str = "query",
    ) -> pd.DataFrame:
        """
        Execute a query with pre-aggregate optimization and automatic fallback.

        This method encapsulates the common pattern of trying pre-aggregate queries first,
        then falling back to raw data queries if the pre-aggregate fails.

        :param pre_aggregate_manager: Manager for pre-aggregate operations
        :param pre_agg_query_builder: Function that returns pre-aggregate query string
        :param raw_query_builder: Function that returns raw data query string
        :param interval: Time interval for aggregation
        :param agg_funcs: List of aggregation functions
        :param column_mapping_rules: Rules for mapping column names in pre-aggregate results
        :param debug_name: Name for debugging/logging purposes
        :return: DataFrame with query results
        """
        # Import locally to avoid circular dependency
        from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_dataframe_processor import (
            TimescaleDBDataFrameProcessor,
        )

        df_processor = TimescaleDBDataFrameProcessor()

        if pre_aggregate_manager.can_use_pre_aggregates(
            interval=interval, agg_funcs=agg_funcs
        ):
            try:
                # Try pre-aggregate query first
                query = pre_agg_query_builder()
                result = self.run(query=query)
                df = df_processor.from_query_result(result)

                if not df.empty and column_mapping_rules:
                    # Apply flexible column mapping for pre-aggregate results
                    mapping = df_processor.build_flexible_column_mapping(
                        df, column_mapping_rules
                    )
                    df = df_processor.apply_column_mapping(df, mapping)

                return df

            except Exception as e:
                logger.warning(
                    f"Pre-aggregate {debug_name} query failed, falling back to raw data",
                    error=mlrun.errors.err_to_str(e),
                )

        # Fallback to raw data query
        raw_query = raw_query_builder()
        result = self.run(query=raw_query)
        return df_processor.from_query_result(result)

    def _execute_with_retry(
        self,
        cursor_operation_callable: Callable[
            [psycopg.Cursor[Any]], Optional[QueryResult]
        ],
        operation_name: str,
    ) -> Optional[QueryResult]:
        """
        Generic retry wrapper for database operations.

        PostgreSQL Error Handling Strategy Matrix (Currently Implemented):

        | Category                    |Retry?| Timing           | Reason                           |
        |-----------------------------|------|------------------|----------------------------------|
        | DeadlockDetected            |  Yes | 0.1s, 0.2s, 0.4s | Auto-rollback, fast resolution  |
        | Other OperationalError      |  Yes | 1s, 2s, 4s       | Network/server recovery time     |
        | InterfaceError              |  Yes | 1s, 2s, 4s       | Client connection issues         |
        | All Other psycopg.Error     |  No  | -                | Pass through without wrapping    |

        Note: PostgreSQL automatically rolls back failed transactions, so explicit
        rollback is only needed for DeadlockDetected where we retry the operation.

        Note: Unhandled errors are passed through without wrapping to preserve
        original exception types and stack traces for proper debugging.

        :param cursor_operation_callable: Function that takes a cursor and executes the operation
        :param operation_name: Name for logging (e.g., "statements", "query")
        :return: Result of cursor_operation_callable()
        """
        deadlock_attempts = 0
        connection_attempts = 0

        while True:
            try:
                # Execute operation within a transaction
                with self.pool.connection() as conn:
                    conn.autocommit = self._autocommit
                    with conn.cursor() as cursor:
                        result = cursor_operation_callable(cursor)
                        if not self._autocommit:
                            conn.commit()
                        return result
            except (psycopg.OperationalError, psycopg.InterfaceError) as e:
                # Different retry limits and timing based on error type
                if isinstance(e, psycopg.errors.DeadlockDetected):
                    if deadlock_attempts >= self.MAX_DEADLOCK_RETRIES:
                        raise mlrun.errors.MLRunRuntimeError(
                            f"Database {operation_name} failed: deadlock persisted "
                            f"after {self.MAX_DEADLOCK_RETRIES} retries: {e}"
                        ) from e
                    # Fast retry for deadlocks: ~0.1s, ~0.2s, ~0.4s with jitter
                    delay = (2**deadlock_attempts) * 0.1 + random.uniform(0, 0.05)
                    error_type = "deadlock"
                    deadlock_attempts += 1
                else:
                    if connection_attempts >= self._max_retries:
                        raise mlrun.errors.MLRunRuntimeError(
                            f"Database {operation_name} failed after "
                            f"{self._max_retries} connection retries: {e}"
                        ) from e
                    # Slower retry for connection issues: 1s, 2s, 4s
                    delay = self._retry_delay * (2**connection_attempts)
                    error_type = "connection"
                    connection_attempts += 1

                logger.warning(
                    f"TimescaleDB {error_type} error in {operation_name}, retrying",
                    attempt=deadlock_attempts
                    if error_type == "deadlock"
                    else connection_attempts,
                    max_retries=self.MAX_DEADLOCK_RETRIES
                    if error_type == "deadlock"
                    else self._max_retries,
                    delay=delay,
                    error=mlrun.errors.err_to_str(e),
                )
                time.sleep(delay)
