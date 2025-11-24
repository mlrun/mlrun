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

from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
import v3io_frames.client

import mlrun
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors
import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_schema as timescaledb_schema
import mlrun.utils
from mlrun.common.schemas.model_monitoring.model_endpoints import _MetricPoint
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    Statement,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_dataframe_processor import (
    TimescaleDBDataFrameProcessor,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_query_builder import (
    TimescaleDBQueryBuilder,
)
from mlrun.model_monitoring.helpers import get_invocations_fqn


class TimescaleDBPredictionsQueries:
    """
    Query class containing predictions-related query methods for TimescaleDB.

    Can be used as a mixin or standalone instance with proper initialization.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        connection=None,
        pre_aggregate_manager=None,
        tables: Optional[dict] = None,
    ):
        """
        Initialize TimescaleDB predictions query handler.

        :param project: Project name
        :param connection: TimescaleDB connection instance
        :param pre_aggregate_manager: PreAggregateManager instance
        :param tables: Dictionary of table schemas
        """
        self.project = project
        self._connection = connection
        self._pre_aggregate_manager = pre_aggregate_manager
        self.tables = tables

    def read_predictions_impl(
        self,
        *,
        endpoint_id: Optional[str] = None,
        start: datetime,
        end: datetime,
        columns: Optional[list[str]] = None,
        aggregation_window: Optional[str] = None,
        agg_funcs: Optional[list[str]] = None,
        limit: Optional[int] = None,
        use_pre_aggregates: bool = True,
    ) -> pd.DataFrame:
        """Read predictions data from TimescaleDB (predictions table) - returns DataFrame.

        :param endpoint_id: Endpoint ID to filter by, or None to get all endpoints
        :param start: Start time
        :param end: End time
        :param columns: Optional list of specific columns to return
        :param aggregation_window: Optional aggregation window (e.g., "1h", "1d")
        :param agg_funcs: Optional list of aggregation functions (e.g., ["avg", "max"])
        :param limit: Optional limit on number of results
        :param use_pre_aggregates: Whether to use pre-aggregates if available
        :return: DataFrame with predictions data
        """
        if (agg_funcs and not aggregation_window) or (
            aggregation_window and not agg_funcs
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "both or neither of `aggregation_window` and `agg_funcs` must be provided"
            )

        # Align times if aggregation window is provided
        start, end = self._pre_aggregate_manager.align_time_range(
            start, end, aggregation_window
        )

        # Check if we can use pre-aggregates
        can_use_pre_aggregates = (
            use_pre_aggregates
            and self._pre_aggregate_manager.can_use_pre_aggregates(
                interval=aggregation_window, agg_funcs=agg_funcs
            )
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]
        filter_query = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_id)

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
            interval=aggregation_window if can_use_pre_aggregates else None,
            agg_funcs=agg_funcs if can_use_pre_aggregates else None,
            limit=limit,
            use_pre_aggregates=can_use_pre_aggregates,
        )

        result = self._connection.run(query=query)
        df = TimescaleDBDataFrameProcessor.from_query_result(result)

        if not df.empty:
            # Set up time index based on whether we used aggregation
            if aggregation_window and can_use_pre_aggregates:
                time_col = timescaledb_schema.TIME_BUCKET_COLUMN
            else:
                time_col = table_schema.time_column

            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)

        return df

    def read_predictions(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        aggregation_window: Optional[str] = None,
        agg_funcs: Optional[list[str]] = None,
        limit: Optional[int] = None,
        use_pre_aggregates: bool = True,
    ) -> Union[
        mm_schemas.ModelEndpointMonitoringMetricValues,
        mm_schemas.ModelEndpointMonitoringMetricNoData,
    ]:
        """Read predictions with optional pre-aggregate optimization."""

        table_schema = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]
        columns = [
            table_schema.time_column,
            mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT,
        ]

        # Get raw DataFrame from read_predictions_impl
        df = self.read_predictions_impl(
            endpoint_id=endpoint_id,
            start=start,
            end=end,
            columns=columns,
            aggregation_window=aggregation_window,
            agg_funcs=agg_funcs,
            limit=limit,
            use_pre_aggregates=use_pre_aggregates,
        )

        # Convert to domain objects
        full_name = get_invocations_fqn(self.project)

        if df.empty:
            return TimescaleDBDataFrameProcessor.handle_empty_dataframe(full_name)

        # Determine value column name based on whether aggregation was used
        can_use_pre_aggregates = (
            use_pre_aggregates
            and aggregation_window
            and agg_funcs
            and self._pre_aggregate_manager.can_use_pre_aggregates(
                interval=aggregation_window, agg_funcs=agg_funcs
            )
        )

        if agg_funcs and can_use_pre_aggregates:
            value_col = (
                f"{agg_funcs[0]}_{mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT}"
            )
        else:
            value_col = mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT

        return mm_schemas.ModelEndpointMonitoringMetricValues(
            full_name=full_name,
            values=[
                _MetricPoint(timestamp=timestamp, value=value)
                for timestamp, value in zip(df.index, df[value_col])
            ],
        )

    def get_last_request(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get last request timestamp with optional pre-aggregate optimization."""

        # Prepare time range and interval (no auto-determination since interval may be None)
        start, end, interval = TimescaleDBQueryBuilder.prepare_time_range_and_interval(
            self._pre_aggregate_manager,
            start,
            end,
            interval,
            auto_determine_interval=False,
        )
        use_pre_aggregates = self._pre_aggregate_manager.can_use_pre_aggregates(
            interval=interval
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]
        filter_query = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_ids)

        if use_pre_aggregates:
            # Calculate latest (MAX) timestamp and corresponding latency per endpoint
            # Use subquery to get time-bucketed data, then MAX over those results
            subquery = table_schema._get_records_query(
                start=start,
                end=end,
                columns_to_filter=[
                    timescaledb_schema.TIME_BUCKET_COLUMN,
                    f"max_{table_schema.time_column}",
                    f"max_{mm_schemas.EventFieldType.LATENCY}",
                    mm_schemas.WriterEvent.ENDPOINT_ID,
                ],
                filter_query=filter_query,
                agg_funcs=["max"],
                interval=interval,
                use_pre_aggregates=True,
            )

            # Use helper to build endpoint aggregation query
            query = TimescaleDBQueryBuilder.build_endpoint_aggregation_query(
                subquery=subquery,
                aggregation_columns={
                    mm_schemas.EventFieldType.LAST_REQUEST: f"MAX(max_{table_schema.time_column})",
                    "last_latency": f"MAX(max_{mm_schemas.EventFieldType.LATENCY})",
                },
            )

            result = self._connection.run(query=query)
            df = TimescaleDBDataFrameProcessor.from_query_result(result)
        else:
            # Use PostgreSQL DISTINCT ON for raw data - most efficient approach
            query = f"""
            SELECT DISTINCT ON ({mm_schemas.WriterEvent.ENDPOINT_ID})
                {mm_schemas.WriterEvent.ENDPOINT_ID} AS endpoint_id,
                {table_schema.time_column} AS {mm_schemas.EventFieldType.LAST_REQUEST},
                {mm_schemas.EventFieldType.LATENCY} AS last_latency
            FROM {table_schema.full_name()}
            WHERE {filter_query}
            AND {table_schema.time_column} >= '{start}'
            AND {table_schema.time_column} <= '{end}'
            ORDER BY {mm_schemas.WriterEvent.ENDPOINT_ID}, {table_schema.time_column} DESC;
            """

            result = self._connection.run(query=query)
            df = TimescaleDBDataFrameProcessor.from_query_result(result)

        # Convert timestamp to proper format (common for both paths)
        if not df.empty and mm_schemas.EventFieldType.LAST_REQUEST in df.columns:
            df[mm_schemas.EventFieldType.LAST_REQUEST] = pd.to_datetime(
                df[mm_schemas.EventFieldType.LAST_REQUEST], errors="coerce", utc=True
            )

        return df

    def get_avg_latency(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        """Get average latency with automatic pre-aggregate optimization, returning single value per endpoint."""

        # Convert single endpoint to list for consistent handling
        if isinstance(endpoint_ids, str):
            endpoint_ids = [endpoint_ids]

        # Set default start time and get end time
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        # Prepare time range with auto-determined interval
        start, end, interval = TimescaleDBQueryBuilder.prepare_time_range_and_interval(
            self._pre_aggregate_manager, start, end
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]
        filter_query = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_ids)

        def build_pre_agg_query():
            # Calculate overall average in SQL across all time buckets
            # Use subquery to get time-bucketed data, then AVG over those results
            subquery = table_schema._get_records_query(
                start=start,
                end=end,
                columns_to_filter=[
                    timescaledb_schema.TIME_BUCKET_COLUMN,
                    mm_schemas.ModelEndpointSchema.AVG_LATENCY,
                    mm_schemas.WriterEvent.ENDPOINT_ID,
                ],
                filter_query=filter_query,
                agg_funcs=["avg"],
                interval=interval,
                use_pre_aggregates=True,
            )

            # Use helper to build endpoint aggregation query
            return TimescaleDBQueryBuilder.build_endpoint_aggregation_query(
                subquery=subquery,
                aggregation_columns={
                    mm_schemas.ModelEndpointSchema.AVG_LATENCY: f"AVG({mm_schemas.ModelEndpointSchema.AVG_LATENCY})"
                },
            )

        def build_raw_query():
            # Single aggregated value across entire time range
            columns = [
                f"{mm_schemas.WriterEvent.ENDPOINT_ID} AS {mm_schemas.WriterEvent.ENDPOINT_ID}",
                f"AVG({mm_schemas.EventFieldType.LATENCY}) AS {mm_schemas.ModelEndpointSchema.AVG_LATENCY}",
            ]
            group_by_columns = [mm_schemas.WriterEvent.ENDPOINT_ID]

            # Add additional filter to exclude invalid latency values
            latency_col = mm_schemas.EventFieldType.LATENCY
            latency_filter = f"{latency_col} IS NOT NULL AND {latency_col} > 0"
            enhanced_filter_query = (
                f"{filter_query} AND {latency_filter}"
                if filter_query
                else latency_filter
            )

            return table_schema._get_records_query(
                start=start,
                end=end,
                columns_to_filter=columns,
                filter_query=enhanced_filter_query,
                group_by=group_by_columns,
                order_by=mm_schemas.WriterEvent.ENDPOINT_ID,
            )

        # Column mapping rules for results (both pre-agg and raw return same structure now)
        column_mapping_rules = {
            mm_schemas.ModelEndpointSchema.AVG_LATENCY: [
                mm_schemas.ModelEndpointSchema.AVG_LATENCY,
                "average_latency",
                mm_schemas.EventFieldType.LATENCY,
            ],
            mm_schemas.WriterEvent.ENDPOINT_ID: [mm_schemas.WriterEvent.ENDPOINT_ID],
        }

        # Both queries now return single value per endpoint, no post-processing needed
        return self._connection.execute_with_fallback(
            self._pre_aggregate_manager,
            build_pre_agg_query,
            build_raw_query,
            interval=interval,
            agg_funcs=["avg"],
            column_mapping_rules=column_mapping_rules,
            debug_name="avg_latency",
        )

    def count_processed_model_endpoints(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        application_names: Optional[Union[str, list[str]]] = None,
    ) -> dict[str, int]:
        """
        Optimized count with application filtering using JOIN approach.

        This implementation:
        1. Uses JOIN when application filtering is needed (most performant)
        2. Falls back to simple query when no filtering (fastest for that case)
        3. Leverages TimescaleDB's chunk exclusion and parallel processing
        4. Can utilize pre-aggregates when available
        """
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._pre_aggregate_manager.get_start_end(start, end)

        predictions_table = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]

        if application_names:
            # Ensure application_names is a list
            if isinstance(application_names, str):
                application_names = [application_names]

            result = {}

            # For each application, call the existing JOIN method and wrap result in dict
            for app_name in application_names:
                # Use existing _count_with_application_join but extract count for single app
                count = self._count_with_application_join(
                    predictions_table,
                    start,
                    end,
                    [app_name],  # Pass as list to existing method
                )
                result[app_name] = count

            return result
        else:
            # Use existing simple count method and wrap result
            total_count = self._count_simple(predictions_table, start, end)
            return {"total": total_count} if total_count > 0 else {}

    def _count_with_application_join(
        self,
        predictions_table,
        start: datetime,
        end: datetime,
        application_names: Union[str, list[str]],
    ) -> int:
        """
        Use JOIN with metrics table for application filtering.

        Performance characteristics:
        - Leverages indexes on both tables
        - TimescaleDB optimizes time-based JOINs
        - Chunk exclusion works on both sides
        - DISTINCT applied after filtering
        """
        metrics_table = self.tables[mm_schemas.TimescaleDBTables.METRICS]

        # Normalize application_names to list for consistent handling
        if isinstance(application_names, str):
            app_names_list = [application_names]
        else:
            app_names_list = list(application_names)

        # Build parameterized query with proper placeholders
        app_placeholders = ", ".join(["%s"] * len(app_names_list))

        query_sql = f"""
        SELECT COUNT(DISTINCT p.{mm_schemas.WriterEvent.ENDPOINT_ID}) AS endpoint_count
        FROM {predictions_table.full_name()} p
        INNER JOIN {metrics_table.full_name()} m
            ON p.{mm_schemas.WriterEvent.ENDPOINT_ID} = m.{mm_schemas.WriterEvent.ENDPOINT_ID}
            AND m.{metrics_table.time_column} >= %s
            AND m.{metrics_table.time_column} <= %s
        WHERE p.{predictions_table.time_column} >= %s
            AND p.{predictions_table.time_column} <= %s
            AND m.{mm_schemas.WriterEvent.APPLICATION_NAME} IN ({app_placeholders})
        """

        # Parameters: [start, end, start, end] + application_names_list
        params = [start, end, start, end] + app_names_list

        stmt = Statement(query_sql, params)
        result = self._connection.run(query=stmt)

        return result.data[0][0] if result and result.data else 0

    def _count_simple(self, predictions_table, start: datetime, end: datetime) -> int:
        """
        Simple count without application filtering.

        Uses the schema's query builder for consistency and potential pre-aggregate usage.
        """
        columns = [
            f"COUNT(DISTINCT {mm_schemas.WriterEvent.ENDPOINT_ID}) AS endpoint_count"
        ]

        query = predictions_table._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
        )

        result = self._connection.run(query=query)
        return result.data[0][0] if result and result.data else 0
