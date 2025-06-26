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
from threading import Lock
from typing import Optional, Union

import psycopg2
import psycopg2.pool

import mlrun.errors


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


# Global connection pool and lock (similar to TDEngine pattern)
_connection_pool = None
_connection_lock = Lock()


class TimescaleDBConnection:
    """
    TimescaleDB connection with shared connection pool and parameterized query support.

    Features:
    - Shared connection pool among all threads for optimal resource usage
    - Thread-safe connection borrowing/returning
    - Automatic connection reuse across threads
    - Configurable pool size based on expected thread load
    - Robust retry logic with connection recovery
    - Exponential backoff for transient failures
    - Support for parameterized queries
    """

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

    def pool(self) -> psycopg2.pool.ThreadedConnectionPool:
        """Get or create the shared connection pool (thread-safe singleton pattern)."""
        global _connection_pool

        if _connection_pool:
            return _connection_pool

        with _connection_lock:
            if not _connection_pool:
                _connection_pool = self._create_pool()

        return _connection_pool

    def _create_pool(self) -> psycopg2.pool.ThreadedConnectionPool:
        """Create a new shared connection pool."""
        try:
            return psycopg2.pool.ThreadedConnectionPool(
                minconn=self._min_connections,
                maxconn=self._max_connections,
                dsn=self._dsn,
            )
        except psycopg2.Error as e:
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed to create connection pool: {e}"
            ) from e

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
        statements = self._normalize_statements(statements)

        for attempt in range(self._max_retries + 1):
            try:
                return self._execute_operation(statements, query)
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                if attempt < self._max_retries:
                    self._handle_retry(attempt)
                else:
                    raise mlrun.errors.MLRunRuntimeError(
                        f"Database operation failed after {self._max_retries + 1} attempts: {e}"
                    ) from e
            except psycopg2.Error as e:
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
        if isinstance(statements, (str, Statement)):
            return [statements]
        return statements

    def _execute_operation(
        self,
        statements: list[Union[str, Statement]],
        query: Optional[Union[str, Statement]],
    ) -> Optional[QueryResult]:
        """Execute a single database operation (statements + optional query)."""
        conn = self.pool().getconn()
        conn.autocommit = self._autocommit
        cursor = None

        try:
            cursor = conn.cursor()

            self._execute_statements(cursor, statements)
            if not self._autocommit:
                conn.commit()

            return self._execute_query(cursor, query) if query else None
        except Exception:
            if not self._autocommit:
                with contextlib.suppress(Exception):
                    conn.rollback()
            raise
        finally:
            self._cleanup_connection(conn, cursor)

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
            field_names = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            data = [tuple(row) for row in results]
            return QueryResult(data, field_names)
        else:
            return QueryResult([], [])

    def _handle_retry(self, attempt: int) -> None:
        """Handle retry logic with exponential backoff."""
        wait_time = self._retry_delay * (2**attempt)
        time.sleep(wait_time)

    def _cleanup_connection(self, conn, cursor) -> None:
        """Clean up connection and cursor resources."""
        # Clean up cursor
        if cursor:
            with contextlib.suppress(Exception):
                cursor.close()
        # Return connection to pool if healthy
        if conn and not conn.closed:
            try:
                self.pool().putconn(conn)
            except Exception:
                # If putconn fails, just close the connection
                with contextlib.suppress(Exception):
                    conn.close()
