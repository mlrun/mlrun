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

import datetime
from dataclasses import dataclass
from io import StringIO
from typing import Optional

import mlrun.common.schemas.model_monitoring as mm_schemas
from mlrun.model_monitoring.db.tsdb.preaggregate import PreAggregateConfig
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_query_builder import (
    TimescaleDBNaming,
)

_MODEL_MONITORING_SCHEMA = "mlrun_model_monitoring"

# TimescaleDB-specific constants
TIME_BUCKET_COLUMN = "time_bucket"

# Database schema constants
MODEL_ERROR_MAX_LENGTH = 1000
CUSTOM_METRICS_MAX_LENGTH = 1000
RESULT_EXTRA_DATA_MAX_LENGTH = 1000


def create_table_schemas(project: str) -> dict:
    """Create all TimescaleDB table schemas for a project.

    This consolidated function eliminates duplication across connector, operations, and test fixtures.

    Args:
        project: The project name for table creation

    Returns:
        Dictionary mapping TimescaleDBTables enum values to table schema objects
    """
    import mlrun

    schema = f"{_MODEL_MONITORING_SCHEMA}_{mlrun.mlconf.system_id}"
    return {
        mm_schemas.TimescaleDBTables.APP_RESULTS: AppResultTable(
            project=project, schema=schema
        ),
        mm_schemas.TimescaleDBTables.METRICS: Metrics(project=project, schema=schema),
        mm_schemas.TimescaleDBTables.PREDICTIONS: Predictions(
            project=project, schema=schema
        ),
        mm_schemas.TimescaleDBTables.ERRORS: Errors(project=project, schema=schema),
    }


class _TimescaleDBColumnType:
    """Represents a TimescaleDB column type with optional constraints."""

    def __init__(
        self, data_type: str, length: Optional[int] = None, nullable: bool = True
    ):
        self.data_type = data_type
        self.length = length
        self.nullable = nullable

    def __str__(self):
        if self.length is not None:
            return f"{self.data_type}({self.length})"
        else:
            return self.data_type


@dataclass
class TimescaleDBSchema:
    """
    A class to represent a hypertable schema in TimescaleDB. Using this schema, you can generate the relevant queries to
    create, insert, delete and query data from TimescaleDB. At the moment, there are 4 schemas: AppResultTable,
    Metrics, Predictions, and Errors.
    """

    def __init__(
        self,
        table_name: str,
        columns: dict[str, _TimescaleDBColumnType],
        time_column: str,
        project: str,
        schema: Optional[str] = None,
        chunk_time_interval: str = "1 day",
        indexes: Optional[list[str]] = None,
    ):
        self.table_name = f"{table_name}_{project.replace('-', '_')}"
        self.columns = columns
        self.time_column = time_column
        self.schema = schema or _MODEL_MONITORING_SCHEMA
        self.chunk_time_interval = chunk_time_interval
        self.indexes = indexes or []
        self.project = project

    def full_name(self) -> str:
        """Return the fully qualified table name (schema.table_name)."""
        return f"{self.schema}.{self.table_name}"

    def _create_table_query(self) -> str:
        """Create the base table SQL."""
        columns_def = ", ".join(
            f"{col} {col_type}" + ("" if col_type.nullable else " NOT NULL")
            for col, col_type in self.columns.items()
        )
        return f"CREATE TABLE IF NOT EXISTS {self.full_name()} ({columns_def});"

    def _create_hypertable_query(self) -> str:
        """Convert table to hypertable."""
        return (
            f"SELECT create_hypertable('{self.full_name()}', '{self.time_column}', "
            f"chunk_time_interval => INTERVAL '{self.chunk_time_interval}', if_not_exists => TRUE);"
        )

    def _create_indexes_query(self) -> list[str]:
        """Create indexes for the table."""
        queries = []
        for index_columns in self.indexes:
            index_name = f"idx_{self.table_name}_{index_columns.replace(',', '_').replace(' ', '_')}"
            queries.append(
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f"ON {self.full_name()} ({index_columns});"
            )
        return queries

    def _create_pre_aggregate_tables_query(
        self, config: PreAggregateConfig
    ) -> list[str]:
        """Create pre-aggregate tables for each interval."""
        queries = []

        for interval in config.aggregate_intervals:
            agg_table_name = TimescaleDBNaming.get_agg_table_name(
                self.table_name, interval
            )

            # Create aggregate table structure
            agg_columns = [f"{TIME_BUCKET_COLUMN} TIMESTAMPTZ NOT NULL"]

            # Add aggregated columns for numeric fields
            for col, col_type in self.columns.items():
                if col == self.time_column:
                    continue
                if col_type.data_type in ["DOUBLE PRECISION", "INTEGER", "BIGINT"]:
                    agg_columns.extend(
                        f"{func}_{col} {col_type}" for func in config.agg_functions
                    )
                else:
                    # For non-numeric columns, keep the original type for grouping
                    agg_columns.append(f"{col} {col_type}")

            create_agg_table = f"CREATE TABLE IF NOT EXISTS {self.schema}.{agg_table_name} ({', '.join(agg_columns)});"

            # Create hypertable for aggregate table
            create_agg_hypertable = (
                f"SELECT create_hypertable('{self.schema}.{agg_table_name}', "
                f"'{TIME_BUCKET_COLUMN}', chunk_time_interval => INTERVAL "
                f"'{self._get_chunk_interval_for_agg(interval)}', if_not_exists => TRUE);"
            )

            queries.extend([create_agg_table, create_agg_hypertable])

        return queries

    def _get_chunk_interval_for_agg(self, interval: str) -> str:
        """Get appropriate chunk interval for aggregate tables."""
        interval_to_chunk = {
            "10m": "1 hour",
            "1h": "1 day",
            "6h": "1 day",
            "1d": "7 days",
            "1w": "1 month",
            "1M": "3 months",
        }
        return interval_to_chunk.get(interval, "1 day")

    def _create_continuous_aggregates_query(
        self, config: PreAggregateConfig
    ) -> list[str]:
        """Create TimescaleDB continuous aggregates for pre-computation."""
        queries = []

        for interval in config.aggregate_intervals:
            cagg_name = TimescaleDBNaming.get_cagg_view_name(self.table_name, interval)

            # Build SELECT clause for continuous aggregate
            select_parts = [
                f"time_bucket(INTERVAL '{interval}', {self.time_column}) AS {TIME_BUCKET_COLUMN}"
            ]

            # Add aggregations for numeric columns
            for col, col_type in self.columns.items():
                if col == self.time_column:
                    continue
                if col_type.data_type in ["DOUBLE PRECISION", "INTEGER", "BIGINT"]:
                    for func in config.agg_functions:
                        if func == "count":
                            select_parts.append(f"COUNT({col}) AS {func}_{col}")
                        else:
                            select_parts.append(
                                f"{func.upper()}({col}) AS {func}_{col}"
                            )
                elif col in [
                    mm_schemas.WriterEvent.ENDPOINT_ID,
                    mm_schemas.WriterEvent.APPLICATION_NAME,
                    mm_schemas.MetricData.METRIC_NAME,
                    mm_schemas.ResultData.RESULT_NAME,
                ]:
                    select_parts.append(col)

            # Group by clause
            group_by_cols = [TIME_BUCKET_COLUMN]
            for col in self.columns:
                if col == self.time_column:
                    continue
                if col in [
                    mm_schemas.WriterEvent.ENDPOINT_ID,
                    mm_schemas.WriterEvent.APPLICATION_NAME,
                    mm_schemas.MetricData.METRIC_NAME,
                    mm_schemas.ResultData.RESULT_NAME,
                ]:
                    group_by_cols.append(col)

            create_cagg = (
                f"CREATE MATERIALIZED VIEW IF NOT EXISTS {self.schema}.{cagg_name} "
                f"WITH (timescaledb.continuous) "
                f"AS SELECT {', '.join(select_parts)} FROM {self.full_name()} "
                f"GROUP BY {', '.join(group_by_cols)} WITH NO DATA;"
            )

            queries.append(create_cagg)

        return queries

    def _create_retention_policies_query(self, config: PreAggregateConfig) -> list[str]:
        """Create retention policies for tables."""
        queries = []

        # Retention for main table
        if "raw" in config.retention_policy:
            queries.append(
                f"SELECT add_retention_policy('{self.full_name()}', INTERVAL "
                f"'{config.retention_policy['raw']}', if_not_exists => TRUE);"
            )

        # Retention for continuous aggregates
        for interval in config.aggregate_intervals:
            if interval in config.retention_policy:
                cagg_name = TimescaleDBNaming.get_cagg_view_name(
                    self.table_name, interval
                )
                queries.append(
                    f"SELECT add_retention_policy('{self.schema}.{cagg_name}', INTERVAL "
                    f"'{config.retention_policy[interval]}', if_not_exists => TRUE);"
                )

        return queries

    def drop_table_query(self) -> str:
        """Drop the main table."""
        return f"DROP TABLE IF EXISTS {self.full_name()} CASCADE;"

    def _get_records_query(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        columns_to_filter: Optional[list[str]] = None,
        filter_query: Optional[str] = None,
        interval: Optional[str] = None,
        limit: Optional[int] = None,
        agg_funcs: Optional[list] = None,
        order_by: Optional[str] = None,
        desc: Optional[bool] = None,
        use_pre_aggregates: bool = True,
        group_by: Optional[list[str]] = None,
    ) -> str:
        """Build query to get records from the table or its pre-aggregates."""

        # Determine table to query
        table_name = self.table_name
        time_col = self.time_column

        if interval and agg_funcs and use_pre_aggregates:
            # Use continuous aggregate if available
            table_name = TimescaleDBNaming.get_cagg_view_name(self.table_name, interval)
            time_col = TIME_BUCKET_COLUMN

        with StringIO() as query:
            query.write("SELECT ")

            if columns_to_filter:
                if interval and agg_funcs and use_pre_aggregates:
                    # For pre-aggregates, use column names as-is since they should already be
                    # the correct names from the continuous aggregate view
                    modified_columns = []
                    for col in columns_to_filter:
                        if col == self.time_column:
                            modified_columns.append(TIME_BUCKET_COLUMN)
                        else:
                            # Use column name as-is - caller should provide correct pre-agg column names
                            modified_columns.append(col)
                    query.write(", ".join(modified_columns))
                else:
                    query.write(", ".join(columns_to_filter))
            else:
                query.write("*")

            query.write(f" FROM {self.schema}.{table_name}")

            # WHERE clause
            conditions = []
            if filter_query:
                conditions.append(filter_query)
            if start:
                conditions.append(f"{time_col} >= '{start}'")
            if end:
                conditions.append(f"{time_col} <= '{end}'")

            if conditions:
                query.write(" WHERE " + " AND ".join(conditions))

            # GROUP BY clause (must come before ORDER BY)
            if group_by:
                query.write(f" GROUP BY {', '.join(group_by)}")

            # ORDER BY clause (must come after GROUP BY)
            if order_by:
                direction = " DESC" if desc else " ASC"
                query.write(f" ORDER BY {order_by}{direction}")

            if limit:
                query.write(f" LIMIT {limit}")

            query.write(";")

            return query.getvalue()


@dataclass
class AppResultTable(TimescaleDBSchema):
    """Schema for application results table."""

    def __init__(self, project: str, schema: Optional[str] = None):
        table_name = mm_schemas.TimescaleDBTables.APP_RESULTS
        columns = {
            mm_schemas.WriterEvent.END_INFER_TIME: _TimescaleDBColumnType(
                "TIMESTAMPTZ"
            ),
            mm_schemas.WriterEvent.START_INFER_TIME: _TimescaleDBColumnType(
                "TIMESTAMPTZ"
            ),
            mm_schemas.ResultData.RESULT_VALUE: _TimescaleDBColumnType(
                "DOUBLE PRECISION"
            ),
            mm_schemas.ResultData.RESULT_STATUS: _TimescaleDBColumnType("INTEGER"),
            mm_schemas.ResultData.RESULT_EXTRA_DATA: _TimescaleDBColumnType(
                "VARCHAR", RESULT_EXTRA_DATA_MAX_LENGTH
            ),
            mm_schemas.WriterEvent.ENDPOINT_ID: _TimescaleDBColumnType("VARCHAR", 64),
            mm_schemas.WriterEvent.APPLICATION_NAME: _TimescaleDBColumnType(
                "VARCHAR", 64
            ),
            mm_schemas.ResultData.RESULT_NAME: _TimescaleDBColumnType("VARCHAR", 64),
            mm_schemas.ResultData.RESULT_KIND: _TimescaleDBColumnType("INTEGER"),
        }
        indexes = [
            mm_schemas.WriterEvent.ENDPOINT_ID,
            f"{mm_schemas.WriterEvent.APPLICATION_NAME}, {mm_schemas.ResultData.RESULT_NAME}",
            mm_schemas.WriterEvent.END_INFER_TIME,
        ]
        super().__init__(
            table_name=table_name,
            columns=columns,
            time_column=mm_schemas.WriterEvent.END_INFER_TIME,
            schema=schema,
            project=project,
            indexes=indexes,
        )


@dataclass
class Metrics(TimescaleDBSchema):
    """Schema for metrics table."""

    def __init__(self, project: str, schema: Optional[str] = None):
        table_name = mm_schemas.TimescaleDBTables.METRICS
        columns = {
            mm_schemas.WriterEvent.END_INFER_TIME: _TimescaleDBColumnType(
                "TIMESTAMPTZ"
            ),
            mm_schemas.WriterEvent.START_INFER_TIME: _TimescaleDBColumnType(
                "TIMESTAMPTZ"
            ),
            mm_schemas.MetricData.METRIC_VALUE: _TimescaleDBColumnType(
                "DOUBLE PRECISION"
            ),
            mm_schemas.WriterEvent.ENDPOINT_ID: _TimescaleDBColumnType("VARCHAR", 64),
            mm_schemas.WriterEvent.APPLICATION_NAME: _TimescaleDBColumnType(
                "VARCHAR", 64
            ),
            mm_schemas.MetricData.METRIC_NAME: _TimescaleDBColumnType("VARCHAR", 64),
        }
        indexes = [
            mm_schemas.WriterEvent.ENDPOINT_ID,
            f"{mm_schemas.WriterEvent.APPLICATION_NAME}, {mm_schemas.MetricData.METRIC_NAME}",
            mm_schemas.WriterEvent.END_INFER_TIME,
            f"{mm_schemas.WriterEvent.END_INFER_TIME}, {mm_schemas.WriterEvent.ENDPOINT_ID},\
                        {mm_schemas.WriterEvent.APPLICATION_NAME}",
            f"{mm_schemas.WriterEvent.APPLICATION_NAME}, {mm_schemas.WriterEvent.END_INFER_TIME}",
        ]
        super().__init__(
            table_name=table_name,
            columns=columns,
            time_column=mm_schemas.WriterEvent.END_INFER_TIME,
            schema=schema,
            project=project,
            indexes=indexes,
        )


@dataclass
class Predictions(TimescaleDBSchema):
    """Schema for predictions table."""

    def __init__(self, project: str, schema: Optional[str] = None):
        table_name = mm_schemas.TimescaleDBTables.PREDICTIONS
        columns = {
            mm_schemas.WriterEvent.END_INFER_TIME: _TimescaleDBColumnType(
                "TIMESTAMPTZ"
            ),
            mm_schemas.EventFieldType.LATENCY: _TimescaleDBColumnType(
                "DOUBLE PRECISION"
            ),
            mm_schemas.EventKeyMetrics.CUSTOM_METRICS: _TimescaleDBColumnType(
                "VARCHAR", CUSTOM_METRICS_MAX_LENGTH
            ),
            mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT: _TimescaleDBColumnType(
                "DOUBLE PRECISION"
            ),
            mm_schemas.EventFieldType.EFFECTIVE_SAMPLE_COUNT: _TimescaleDBColumnType(
                "INTEGER"
            ),
            mm_schemas.WriterEvent.ENDPOINT_ID: _TimescaleDBColumnType("VARCHAR", 64),
        }

        indexes = [
            mm_schemas.WriterEvent.ENDPOINT_ID,
            mm_schemas.WriterEvent.END_INFER_TIME,
            f"{mm_schemas.WriterEvent.END_INFER_TIME}, {mm_schemas.WriterEvent.ENDPOINT_ID}",
        ]
        super().__init__(
            table_name=table_name,
            columns=columns,
            time_column=mm_schemas.WriterEvent.END_INFER_TIME,
            schema=schema,
            project=project,
            indexes=indexes,
        )


@dataclass
class Errors(TimescaleDBSchema):
    """Schema for errors table."""

    def __init__(self, project: str, schema: Optional[str] = None):
        table_name = mm_schemas.TimescaleDBTables.ERRORS
        columns = {
            mm_schemas.EventFieldType.TIME: _TimescaleDBColumnType("TIMESTAMPTZ"),
            mm_schemas.EventFieldType.MODEL_ERROR: _TimescaleDBColumnType(
                "VARCHAR", MODEL_ERROR_MAX_LENGTH
            ),
            mm_schemas.WriterEvent.ENDPOINT_ID: _TimescaleDBColumnType("VARCHAR", 64),
            mm_schemas.EventFieldType.ERROR_TYPE: _TimescaleDBColumnType("VARCHAR", 64),
        }
        indexes = [
            mm_schemas.WriterEvent.ENDPOINT_ID,
            mm_schemas.EventFieldType.ERROR_TYPE,
            mm_schemas.EventFieldType.TIME,
        ]
        super().__init__(
            table_name=table_name,
            columns=columns,
            time_column=mm_schemas.EventFieldType.TIME,
            schema=schema,
            project=project,
            indexes=indexes,
        )
