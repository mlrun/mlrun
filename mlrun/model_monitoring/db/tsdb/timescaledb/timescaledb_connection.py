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
from typing import Callable, Optional, Union

import pandas as pd
import psycopg
from psycopg_pool import ConnectionPool

import mlrun.errors
from mlrun.model_monitoring.db.tsdb.preaggregate import PreAggregateHandler


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

    def _parse_version(self, version_string: str) -> tuple[int, int, int]:
        """Parse TimescaleDB version string into comparable tuple."""
        try:
            # Handle versions like "2.22.0", "2.7.1-dev", etc.
            version_parts = version_string.split("-")[0].split(".")
            major = int(version_parts[0])
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            patch = int(version_parts[2]) if len(version_parts) > 2 else 0
            return (major, minor, patch)
        except (ValueError, IndexError) as e:
            raise mlrun.errors.MLRunRuntimeError(
                f"Invalid TimescaleDB version format: {version_string}"
            ) from e

    def _version_compare(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1, 0, or 1."""
        v1_tuple = self._parse_version(version1)
        v2_tuple = self._parse_version(version2)

        if v1_tuple < v2_tuple:
            return -1
        elif v1_tuple > v2_tuple:
            return 1
        else:
            return 0

    def _check_timescaledb_version(self) -> None:
        """Check TimescaleDB version and raise error if less than 2.7.0."""
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

                    if not result:
                        raise mlrun.errors.MLRunRuntimeError(
                            "TimescaleDB extension is not installed"
                        )

                    self._timescaledb_version = result[0]

                    # Check minimum version (2.7.0+)
                    if (
                        self._version_compare(
                            self._timescaledb_version, self.MIN_TIMESCALEDB_VERSION
                        )
                        < 0
                    ):
                        raise mlrun.errors.MLRunRuntimeError(
                            f"TimescaleDB version {self._timescaledb_version} is not supported. "
                            f"Minimum required version: {self.MIN_TIMESCALEDB_VERSION} "
                            f"(required for finalized continuous aggregates)"
                        )

        except psycopg.Error as e:
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed to check TimescaleDB version: {e}"
            ) from e
        finally:
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
        Execute statements and optionally return query results with retry logic.

        Supports both string SQL and parameterized Statement objects.
        Uses retry parameters configured in constructor for consistent behavior.

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

        statements = self._normalize_statements(statements)

        for attempt in range(self._max_retries + 1):
            try:
                return self._execute_operation(statements, query)
            except (psycopg.OperationalError, psycopg.InterfaceError) as e:
                if attempt < self._max_retries:
                    self._handle_retry(attempt)
                else:
                    raise mlrun.errors.MLRunRuntimeError(
                        f"Database operation failed after {self._max_retries + 1} attempts: {e}"
                    ) from e
            except psycopg.Error as e:
                raise mlrun.errors.MLRunRuntimeError(
                    f"Database operation failed: {e}"
                ) from e

        # Fallback (should never reach here)
        raise mlrun.errors.MLRunRuntimeError(
            "Database operation failed for unknown reason"
        )

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
                try:
                    self._execute_statements(cursor, statements)
                    if not self._autocommit:
                        conn.commit()

                    return self._execute_query(cursor, query) if query else None
                except Exception:
                    if not self._autocommit:
                        conn.rollback()
                    raise

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

    def _handle_retry(self, attempt: int) -> None:
        """Handle retry logic with exponential backoff."""
        wait_time = self._retry_delay * (2**attempt)
        time.sleep(wait_time)

    def execute_with_fallback(
        self,
        pre_aggregate_handler: PreAggregateHandler,
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

        :param pre_aggregate_handler: Handler for pre-aggregate operations
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

        if pre_aggregate_handler.can_use_pre_aggregates(
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
                # Log the fallback (in production, use proper logging)
                print(
                    f"Pre-aggregate {debug_name} query failed, falling back to raw data: {e}"
                )

        # Fallback to raw data query
        raw_query = raw_query_builder()
        result = self.run(query=raw_query)
        return df_processor.from_query_result(result)
