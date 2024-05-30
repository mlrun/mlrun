# Copyright 2024 Iguazio
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

import datetime
from dataclasses import dataclass
from io import StringIO
from typing import Optional, Union

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.common.types

_MODEL_MONITORING_DATABASE = "mlrun_model_monitoring"


class _TDEngineColumnType:
    def __init__(self, data_type: str, length: int = None):
        self.data_type = data_type
        self.length = length

    def __str__(self):
        if self.length is not None:
            return f"{self.data_type}({self.length})"
        else:
            return self.data_type


class _TDEngineColumn(mlrun.common.types.StrEnum):
    TIMESTAMP = _TDEngineColumnType("TIMESTAMP")
    FLOAT = _TDEngineColumnType("FLOAT")
    INT = _TDEngineColumnType("INT")
    BINARY_40 = _TDEngineColumnType("BINARY", 40)
    BINARY_64 = _TDEngineColumnType("BINARY", 64)
    BINARY_10000 = _TDEngineColumnType("BINARY", 10000)


@dataclass
class TDEngineSchema:
    """
    A class to represent a supertable schema in TDengine. Using this schema, you can generate the relevant queries to
    create, insert, delete and query data from TDengine. At the moment, there are 3 schemas: AppResultTable,
    Metrics, and Predictions.
    """

    def __init__(
        self,
        super_table: str,
        columns: dict[str, str],
        tags: dict[str, str],
    ):
        self.super_table = super_table
        self.columns = columns
        self.tags = tags
        self.database = _MODEL_MONITORING_DATABASE

    def _create_super_table_query(self) -> str:
        columns = ", ".join(f"{col} {val}" for col, val in self.columns.items())
        tags = ", ".join(f"{col} {val}" for col, val in self.tags.items())
        return f"CREATE STABLE if NOT EXISTS {self.database}.{self.super_table} ({columns}) TAGS ({tags});"

    def _create_subtable_query(
        self,
        subtable: str,
        values: dict[str, Union[str, int, float, datetime.datetime]],
    ) -> str:
        try:
            values = ", ".join(f"'{values[val]}'" for val in self.tags)
        except KeyError:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"values must contain all tags: {self.tags.keys()}"
            )
        return f"CREATE TABLE if NOT EXISTS {self.database}.{subtable} USING {self.super_table} TAGS ({values});"

    def _insert_subtable_query(
        self,
        subtable: str,
        values: dict[str, Union[str, int, float, datetime.datetime]],
    ) -> str:
        values = ", ".join(f"'{values[val]}'" for val in self.columns)
        return f"INSERT INTO {self.database}.{subtable} VALUES ({values});"

    def _delete_subtable_query(
        self,
        subtable: str,
        values: dict[str, Union[str, int, float, datetime.datetime]],
    ) -> str:
        values = " AND ".join(
            f"{val} LIKE '{values[val]}'" for val in self.tags if val in values
        )
        if not values:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"values must contain at least one tag: {self.tags.keys()}"
            )
        return f"DELETE FROM {self.database}.{subtable} WHERE {values};"

    def _drop_subtable_query(
        self,
        subtable: str,
    ) -> str:
        return f"DROP TABLE if EXISTS {self.database}.{subtable};"

    def _get_subtables_query(
        self,
        values: dict[str, Union[str, int, float, datetime.datetime]],
    ) -> str:
        values = " AND ".join(
            f"{val} LIKE '{values[val]}'" for val in self.tags if val in values
        )
        if not values:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"values must contain at least one tag: {self.tags.keys()}"
            )
        return f"SELECT tbname FROM {self.database}.{self.super_table} WHERE {values};"

    @staticmethod
    def _get_records_query(
        table: str,
        start: datetime,
        end: datetime,
        columns_to_filter: list[str] = None,
        filter_query: Optional[str] = None,
        interval: Optional[str] = None,
        limit: int = 0,
        agg_funcs: Optional[list] = None,
        sliding_window_step: Optional[str] = None,
        timestamp_column: str = "time",
        database: str = _MODEL_MONITORING_DATABASE,
    ) -> str:
        if agg_funcs and not columns_to_filter:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "`columns_to_filter` must be provided when using aggregate functions"
            )

        # if aggregate function or interval is provided, the other must be provided as well
        if interval and not agg_funcs:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "`agg_funcs` must be provided when using interval"
            )

        if sliding_window_step and not interval:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "`interval` must be provided when using sliding window"
            )

        with StringIO() as query:
            query.write("SELECT ")
            if interval:
                query.write("_wstart, _wend, ")
            if agg_funcs:
                query.write(
                    ", ".join(
                        [f"{a}({col})" for a in agg_funcs for col in columns_to_filter]
                    )
                )
            elif columns_to_filter:
                query.write(", ".join(columns_to_filter))
            else:
                query.write("*")
            query.write(f" FROM {database}.{table}")

            if any([filter_query, start, end]):
                query.write(" WHERE ")
                if filter_query:
                    query.write(f"{filter_query} AND ")
                if start:
                    query.write(f"{timestamp_column} >= '{start}'" + " AND ")
                if end:
                    query.write(f"{timestamp_column} <= '{end}'")
            if interval:
                query.write(f" INTERVAL({interval})")
            if sliding_window_step:
                query.write(f" SLIDING({sliding_window_step})")
            if limit:
                query.write(f" LIMIT {limit}")
            query.write(";")
            return query.getvalue()


@dataclass
class AppResultTable(TDEngineSchema):
    super_table = mm_schemas.TDEngineSuperTables.APP_RESULTS
    columns = {
        mm_schemas.WriterEvent.END_INFER_TIME: _TDEngineColumn.TIMESTAMP,
        mm_schemas.WriterEvent.START_INFER_TIME: _TDEngineColumn.TIMESTAMP,
        mm_schemas.ResultData.RESULT_VALUE: _TDEngineColumn.FLOAT,
        mm_schemas.ResultData.RESULT_STATUS: _TDEngineColumn.INT,
        mm_schemas.ResultData.CURRENT_STATS: _TDEngineColumn.BINARY_10000,
    }

    tags = {
        mm_schemas.EventFieldType.PROJECT: _TDEngineColumn.BINARY_64,
        mm_schemas.WriterEvent.ENDPOINT_ID: _TDEngineColumn.BINARY_64,
        mm_schemas.WriterEvent.APPLICATION_NAME: _TDEngineColumn.BINARY_64,
        mm_schemas.ResultData.RESULT_NAME: _TDEngineColumn.BINARY_64,
        mm_schemas.ResultData.RESULT_KIND: _TDEngineColumn.INT,
    }
    database = _MODEL_MONITORING_DATABASE


@dataclass
class Metrics(TDEngineSchema):
    super_table = mm_schemas.TDEngineSuperTables.METRICS
    columns = {
        mm_schemas.WriterEvent.END_INFER_TIME: _TDEngineColumn.TIMESTAMP,
        mm_schemas.WriterEvent.START_INFER_TIME: _TDEngineColumn.TIMESTAMP,
        mm_schemas.MetricData.METRIC_VALUE: _TDEngineColumn.FLOAT,
    }

    tags = {
        mm_schemas.EventFieldType.PROJECT: _TDEngineColumn.BINARY_64,
        mm_schemas.WriterEvent.ENDPOINT_ID: _TDEngineColumn.BINARY_64,
        mm_schemas.WriterEvent.APPLICATION_NAME: _TDEngineColumn.BINARY_64,
        mm_schemas.MetricData.METRIC_NAME: _TDEngineColumn.BINARY_64,
    }
    database = _MODEL_MONITORING_DATABASE


@dataclass
class Predictions(TDEngineSchema):
    super_table = mm_schemas.TDEngineSuperTables.PREDICTIONS
    columns = {
        mm_schemas.EventFieldType.TIME: _TDEngineColumn.TIMESTAMP,
        mm_schemas.EventFieldType.LATENCY: _TDEngineColumn.FLOAT,
        mm_schemas.EventKeyMetrics.CUSTOM_METRICS: _TDEngineColumn.BINARY_10000,
    }
    tags = {
        mm_schemas.EventFieldType.PROJECT: _TDEngineColumn.BINARY_64,
        mm_schemas.WriterEvent.ENDPOINT_ID: _TDEngineColumn.BINARY_64,
    }
    database = _MODEL_MONITORING_DATABASE
