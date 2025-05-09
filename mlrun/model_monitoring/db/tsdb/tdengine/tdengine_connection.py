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

import traceback
from collections.abc import Callable
from enum import Enum
from typing import Any, Final, Optional, Union

import taosws
from taosws import TaosStmt


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


def _run(connection_string, prefix_statements, q, statements, query):
    try:
        conn = taosws.connect(connection_string)

        for statement in prefix_statements + statements:
            if isinstance(statement, Statement):
                prepared_statement = statement.prepare(conn.statement())
                prepared_statement.execute()
            else:
                conn.execute(statement)

        if not query:
            q.put(None)
            return

        res = conn.query(query)

        # taosws.TaosField is not serializable
        fields = [
            Field(field.name(), field.type(), field.bytes()) for field in res.fields
        ]

        q.put(QueryResult(list(res), fields))
    except Exception as e:
        tb = traceback.format_exc()
        q.put(ErrorResult(tb, e))


class TDEngineConnection:
    def __init__(self, connection_string):
        self._connection_string = connection_string
        self.prefix_statements = []

        self._conn = taosws.connect(self._connection_string)

    def run(
        self,
        statements: Optional[Union[str, Statement, list[Union[str, Statement]]]] = None,
        query: Optional[str] = None,
    ) -> Optional[QueryResult]:
        statements = statements or []
        if not isinstance(statements, list):
            statements = [statements]

        for statement in self.prefix_statements + statements:
            if isinstance(statement, Statement):
                try:
                    prepared_statement = statement.prepare(self._conn.statement())
                    prepared_statement.execute()
                except taosws.Error as e:
                    raise TDEngineError(
                        f"Failed to run prepared statement `{self._conn.statement()}`: {e}"
                    ) from e
            else:
                try:
                    self._conn.execute(statement)
                except taosws.Error as e:
                    raise TDEngineError(
                        f"Failed to run statement `{statement}`: {e}"
                    ) from e

        if not query:
            return None

        try:
            res = self._conn.query(query)
        except taosws.Error as e:
            raise TDEngineError(f"Failed to run query `{query}`: {e}") from e

        fields = [
            Field(field.name(), field.type(), field.bytes()) for field in res.fields
        ]

        return QueryResult(list(res), fields)
