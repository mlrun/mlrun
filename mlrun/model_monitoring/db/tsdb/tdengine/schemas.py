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
from typing import Union

import mlrun.common.schemas.model_monitoring as mm_constants
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
            f"{val} like '{values[val]}'" for val in self.tags if val in values
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
            f"{val} like '{values[val]}'" for val in self.tags if val in values
        )
        if not values:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"values must contain at least one tag: {self.tags.keys()}"
            )
        return f"SELECT tbname FROM {self.database}.{self.super_table} WHERE {values};"

    def _get_records_query(
        self,
        table: str,
        columns_to_filter: list[str] = None,
        filter_query: str = "",
        start: str = datetime.datetime.now().astimezone() - datetime.timedelta(hours=1),
        end: str = datetime.datetime.now().astimezone(),
        timestamp_column: str = "time",
    ) -> str:
        with StringIO() as query:
            query.write("SELECT ")
            if columns_to_filter:
                query.write(", ".join(columns_to_filter))
            else:
                query.write("*")
            query.write(f" from {self.database}.{table}")

            if any([filter_query, start, end]):
                query.write(" where ")
                if filter_query:
                    query.write(f"{filter_query} and ")
                if start:
                    query.write(f"{timestamp_column} >= '{start}'" + " and ")
                if end:
                    query.write(f"{timestamp_column} <= '{end}'")
                full_query = query.getvalue()
                if full_query.endswith(" and "):
                    full_query = full_query[:-5]
            return full_query + ";"


@dataclass
class AppResultTable(TDEngineSchema):
    super_table = mm_constants.TDEngineSuperTables.APP_RESULTS
    columns = {
        mm_constants.WriterEvent.END_INFER_TIME: _TDEngineColumn.TIMESTAMP,
        mm_constants.WriterEvent.START_INFER_TIME: _TDEngineColumn.TIMESTAMP,
        mm_constants.ResultData.RESULT_VALUE: _TDEngineColumn.FLOAT,
        mm_constants.ResultData.RESULT_STATUS: _TDEngineColumn.INT,
        mm_constants.ResultData.CURRENT_STATS: _TDEngineColumn.BINARY_10000,
    }

    tags = {
        mm_constants.EventFieldType.PROJECT: _TDEngineColumn.BINARY_64,
        mm_constants.WriterEvent.ENDPOINT_ID: _TDEngineColumn.BINARY_64,
        mm_constants.WriterEvent.APPLICATION_NAME: _TDEngineColumn.BINARY_64,
        mm_constants.ResultData.RESULT_NAME: _TDEngineColumn.BINARY_64,
    }
    database = _MODEL_MONITORING_DATABASE


@dataclass
class Metrics(TDEngineSchema):
    super_table = mm_constants.TDEngineSuperTables.METRICS
    columns = {
        mm_constants.WriterEvent.END_INFER_TIME: _TDEngineColumn.TIMESTAMP,
        mm_constants.WriterEvent.START_INFER_TIME: _TDEngineColumn.TIMESTAMP,
        mm_constants.MetricData.METRIC_VALUE: _TDEngineColumn.FLOAT,
    }

    tags = {
        mm_constants.EventFieldType.PROJECT: _TDEngineColumn.BINARY_64,
        mm_constants.WriterEvent.ENDPOINT_ID: _TDEngineColumn.BINARY_64,
        mm_constants.WriterEvent.APPLICATION_NAME: _TDEngineColumn.BINARY_64,
        mm_constants.MetricData.METRIC_NAME: _TDEngineColumn.BINARY_64,
    }
    database = _MODEL_MONITORING_DATABASE


@dataclass
class Predictions(TDEngineSchema):
    super_table = mm_constants.TDEngineSuperTables.PREDICTIONS
    columns = {
        mm_constants.EventFieldType.TIME: _TDEngineColumn.TIMESTAMP,
        mm_constants.EventFieldType.LATENCY: _TDEngineColumn.FLOAT,
        mm_constants.EventKeyMetrics.CUSTOM_METRICS: _TDEngineColumn.BINARY_10000,
    }
    tags = {
        mm_constants.EventFieldType.PROJECT: _TDEngineColumn.BINARY_64,
        mm_constants.WriterEvent.ENDPOINT_ID: _TDEngineColumn.BINARY_64,
    }
    database = _MODEL_MONITORING_DATABASE
