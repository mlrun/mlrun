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
from collections.abc import Callable
from enum import Enum
from typing import Any, Final, Optional, Union

import taosws
from taosws import TaosStmt

import mlrun
from mlrun.utils import logger


class _StrEnum(str, Enum):
    pass


class TimestampPrecision(_StrEnum):
    ms = "ms"  # milliseconds
    us = "us"  # microseconds
    ns = "ns"  # nanoseconds


_TS_PRECISION_TO_FACTOR_AND_FUNC: Final[
    dict[TimestampPrecision, tuple[int, Callable[[list[int]], taosws.PyColumnView]]]
] = {
    TimestampPrecision.ms: (10**3, taosws.millis_timestamps_to_column),
    TimestampPrecision.us: (10**6, taosws.micros_timestamps_to_column),
    TimestampPrecision.ns: (10**9, taosws.nanos_timestamps_to_column),
}


class QueryResult:
    def __init__(self, data, fields):
        self.data = data
        self.fields = fields

    def __eq__(self, other):
        return self.data == other.data and self.fields == other.fields

    def __repr__(self):
        return f"QueryResult({self.data}, {self.fields})"


class Field:
    def __init__(self, name, type, bytes):
        self.name = name
        self.type = type
        self.bytes = bytes

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.type == other.type
            and self.bytes == other.bytes
        )

    def __repr__(self):
        return f"Field({self.name}, {self.type}, {self.bytes})"


class TDEngineError(Exception):
    pass


class ErrorResult:
    def __init__(self, tb, err):
        self.tb = tb
        self.err = err


def _get_timestamp_column(
    values: list, timestamp_precision: TimestampPrecision
) -> taosws.PyColumnView:
    factor, to_col_func = _TS_PRECISION_TO_FACTOR_AND_FUNC[timestamp_precision]
    timestamps = [round(timestamp.timestamp() * factor) for timestamp in values]
    return to_col_func(timestamps)


def values_to_column(
    values: list,
    column_type: str,
    timestamp_precision: TimestampPrecision = TimestampPrecision.ms,
) -> taosws.PyColumnView:
    if column_type == "TIMESTAMP":
        return _get_timestamp_column(values, timestamp_precision)
    if column_type == "FLOAT":
        return taosws.floats_to_column(values)
    if column_type == "INT":
        return taosws.ints_to_column(values)
    if column_type.startswith("BINARY"):
        return taosws.binary_to_column(values)

    raise NotImplementedError(f"Unsupported column type '{column_type}'")


class Statement:
    def __init__(
        self,
        columns: dict[str, str],
        subtable: str,
        values: dict[str, Any],
        timestamp_precision: str = TimestampPrecision.ms,
    ) -> None:
        self.columns = columns
        self.subtable = subtable
        self.values = values
        self.timestamp_precision = TimestampPrecision[timestamp_precision]

    def prepare(self, statement: TaosStmt) -> TaosStmt:
        question_marks = ", ".join("?" * len(self.columns))
        statement.prepare(f"INSERT INTO ? VALUES ({question_marks});")
        statement.set_tbname(self.subtable)

        bind_params = []

        for col_name, col_type in self.columns.items():
            val = self.values[col_name]
            bind_params.append(
                values_to_column(
                    [val], col_type, timestamp_precision=self.timestamp_precision
                )
            )

        statement.bind_param(bind_params)
        statement.add_batch()
        return statement


class TDEngineConnection:
    def __init__(self, connection_string, max_retries=3, retry_delay=0.5):
        self._connection_string = connection_string
        self.prefix_statements = []
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._conn = self._create_connection()

    def _create_connection(self):
        """Create a new TDEngine connection."""
        return taosws.connect(self._connection_string)

    def _reconnect(self):
        """Close current connection and create a new one."""
        try:
            if hasattr(self, "_conn") and self._conn:
                self._conn.close()
        except Exception as e:
            logger.warning(f"Error closing connection during reconnect: {e}")

        self._conn = self._create_connection()
        logger.info("Successfully reconnected to TDEngine")

    def _execute_with_retry(self, operation, operation_name, *args, **kwargs):
        """
        Execute an operation with retry logic for connection failures.

        :param operation: The function to execute
        :param operation_name: Name of the operation for logging
        :param args: Arguments to pass to the operation
        :param kwargs: Keyword arguments to pass to the operation
        :return: Result of the operation
        """
        last_exception = None

        for attempt in range(self._max_retries + 1):  # +1 for initial attempt
            try:
                return operation(*args, **kwargs)

            except taosws.Error as e:
                last_exception = e

                if attempt < self._max_retries:
                    logger.warning(
                        f"Connection error during {operation_name} "
                        f"(attempt {attempt + 1}/{self._max_retries + 1}): {e}. "
                        f"Retrying in {self._retry_delay} seconds..."
                    )

                    # Wait before retrying
                    time.sleep(self._retry_delay)

                    # Reconnect
                    try:
                        self._reconnect()
                    except Exception as reconnect_error:
                        logger.error(f"Failed to reconnect: {reconnect_error}")
                        if attempt == self._max_retries - 1:
                            # Last attempt, raise the reconnection error
                            raise TDEngineError(
                                f"Failed to reconnect after {operation_name} failure: {reconnect_error}"
                            ) from reconnect_error
                        continue
                else:
                    # Max retries exceeded
                    logger.error(
                        f"Max retries ({self._max_retries}) exceeded for {operation_name}"
                    )
                    break

            except Exception as e:
                # Non-TDEngine error, don't retry
                raise TDEngineError(
                    f"Unexpected error during {operation_name}: {e}"
                ) from e

        # If we get here, all retries failed
        raise TDEngineError(
            f"Failed to {operation_name} after {self._max_retries} retries: {last_exception}"
        ) from last_exception

    def _execute_statement(self, statement):
        """Execute a single statement (string or Statement object)."""
        if isinstance(statement, Statement):
            prepared_statement = statement.prepare(self._conn.statement())
            prepared_statement.execute()
        else:
            self._conn.execute(statement)

    def _execute_query(self, query):
        """Execute a query and return the result."""
        return self._conn.query(query)

    def run(
        self,
        statements: Optional[Union[str, Statement, list[Union[str, Statement]]]] = None,
        query: Optional[str] = None,
    ) -> Optional[QueryResult]:
        statements = statements or []
        if not isinstance(statements, list):
            statements = [statements]

        # Execute all statements with retry logic
        all_statements = self.prefix_statements + statements
        for i, statement in enumerate(all_statements):
            operation_name = f"execute statement {i + 1}/{len(all_statements)}"
            if isinstance(statement, Statement):
                operation_name += " (prepared)"
            else:
                operation_name += f" `{statement}`"

            self._execute_with_retry(self._execute_statement, operation_name, statement)

        if not query:
            return None

        # Execute query with retry logic
        res = self._execute_with_retry(
            self._execute_query, f"execute query `{query}`", query
        )

        # Process results
        fields = [
            Field(field.name(), field.type(), field.bytes()) for field in res.fields
        ]

        return QueryResult(list(res), fields)

    def close(self):
        """Close the connection."""
        try:
            if self._conn:
                self._conn.close()
                logger.debug("TDEngine connection closed")
                self._conn = None
        except Exception as e:
            logger.warning(
                f"Error closing TDEngine connection: {mlrun.errors.err_to_str(e)}"
            )
