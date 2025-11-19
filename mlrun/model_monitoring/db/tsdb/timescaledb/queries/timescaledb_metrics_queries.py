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

from datetime import datetime
from typing import Optional, Union

import pandas as pd

import mlrun
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    Statement,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_query_builder import (
    TimescaleDBQueryBuilder,
)


class TimescaleDBMetricsQueries:
    """
    Query class containing metrics-related query methods for TimescaleDB.

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
        Initialize TimescaleDB metrics query handler.

        :param project: Project name
        :param connection: TimescaleDB connection instance
        :param pre_aggregate_manager: PreAggregateManager instance
        :param tables: Dictionary of table schemas
        """
        self.project = project
        self._connection = connection
        self._pre_aggregate_manager = pre_aggregate_manager
        self.tables = tables

    def get_model_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: list[str],
        start: str,
        end: str,
        interval: Optional[str] = None,
        agg_function: Optional[str] = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Get real-time metrics with optional pre-aggregate optimization."""

        # Validate that metrics are provided
        if not metrics:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Metric names must be provided"
            )

        # Prepare time range with validation and ISO conversion using helper
        start_dt, end_dt, interval = (
            TimescaleDBQueryBuilder.prepare_time_range_with_validation(
                self._pre_aggregate_manager, start, end, interval, agg_function
            )
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]

        # Build query including metric names for filtering and grouping
        columns = [
            table_schema.time_column,
            mm_schemas.MetricData.METRIC_NAME,
            mm_schemas.MetricData.METRIC_VALUE,
        ]

        # Build filters: endpoint + requested metrics
        endpoint_filter = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_id)
        metrics_filter = TimescaleDBQueryBuilder.build_metrics_filter_from_names(
            metrics
        )
        combined_filter = TimescaleDBQueryBuilder.combine_filters(
            [endpoint_filter, metrics_filter]
        )

        # Use fallback pattern for potential pre-aggregate compatibility issues
        def build_pre_agg_query():
            return table_schema._get_records_query(
                start=start_dt,
                end=end_dt,
                columns_to_filter=columns,
                filter_query=combined_filter,
                interval=interval,
                agg_funcs=[agg_function] if agg_function else None,
                use_pre_aggregates=True,
            )

        def build_raw_query():
            return table_schema._get_records_query(
                start=start_dt,
                end=end_dt,
                columns_to_filter=columns,
                filter_query=combined_filter,
            )

        # Column mapping rules for pre-aggregate results (if needed)
        column_mapping_rules = {
            mm_schemas.MetricData.METRIC_NAME: [mm_schemas.MetricData.METRIC_NAME],
            mm_schemas.MetricData.METRIC_VALUE: [mm_schemas.MetricData.METRIC_VALUE],
            table_schema.time_column: [table_schema.time_column],
        }

        df = self._connection.execute_with_fallback(
            self._pre_aggregate_manager,
            build_pre_agg_query,
            build_raw_query,
            interval=interval,
            agg_funcs=[agg_function] if agg_function else None,
            column_mapping_rules=column_mapping_rules,
            debug_name="get_model_endpoint_real_time_metrics",
        )

        # Process DataFrame result into expected format: {metric_name: [(timestamp, value), ...]}
        metrics_data = {metric_name: [] for metric_name in metrics}

        if not df.empty:
            for _, row in df.iterrows():
                timestamp = row[table_schema.time_column]
                metric_name = row[mm_schemas.MetricData.METRIC_NAME]
                value = row[mm_schemas.MetricData.METRIC_VALUE]

                # Only include requested metrics
                if metric_name in metrics_data:
                    metrics_data[metric_name].append(
                        (timestamp.isoformat(), float(value))
                    )

        return metrics_data

    def read_metrics_data_impl(
        self,
        *,
        endpoint_id: Optional[str] = None,
        start: datetime,
        end: datetime,
        metrics: Optional[list[mm_schemas.ModelEndpointMonitoringMetric]] = None,
    ) -> pd.DataFrame:
        """Read metrics data from TimescaleDB (metrics table only) - returns DataFrame.

        :param endpoint_id: Endpoint ID to filter by, or None to get all endpoints
        :param start: Start time
        :param end: End time
        :param metrics: List of metrics to filter by, or None to get all metrics
        :return: DataFrame with metrics data
        """

        table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]
        name_column = mm_schemas.MetricData.METRIC_NAME
        value_column = mm_schemas.MetricData.METRIC_VALUE
        columns = [
            table_schema.time_column,
            mm_schemas.WriterEvent.APPLICATION_NAME,
            name_column,
            value_column,
        ]

        # Build metrics condition using query builder utilities (accepts None)
        metrics_condition = TimescaleDBQueryBuilder.build_metrics_filter(metrics)
        endpoint_filter = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_id)

        # Combine filters using query builder utilities
        filters = [endpoint_filter, metrics_condition]
        filter_query = TimescaleDBQueryBuilder.combine_filters(filters)

        # Use shared utility for consistent query building with fallback
        df = TimescaleDBQueryBuilder.build_read_data_with_fallback(
            connection=self._connection,
            pre_aggregate_manager=self._pre_aggregate_manager,
            table_schema=table_schema,
            start=start,
            end=end,
            columns=columns,
            filter_query=filter_query,
            name_column=name_column,
            value_column=value_column,
            debug_name="read_metrics_data",
        )

        if not df.empty:
            df[table_schema.time_column] = pd.to_datetime(df[table_schema.time_column])
            df.set_index(table_schema.time_column, inplace=True)

        return df

    def get_metrics_metadata(
        self,
        endpoint_id: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get metrics metadata with optional pre-aggregate optimization."""

        # Prepare time range and interval (no auto-determination since interval passed in)
        start, end, interval = TimescaleDBQueryBuilder.prepare_time_range_and_interval(
            self._pre_aggregate_manager,
            start,
            end,
            interval,
            auto_determine_interval=False,
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]
        filter_query = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_id)

        columns = [
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.MetricData.METRIC_NAME,
            mm_schemas.WriterEvent.ENDPOINT_ID,
        ]

        # Use fallback pattern for potential pre-aggregate compatibility issues
        def build_pre_agg_query():
            return table_schema._get_records_query(
                start=start,
                end=end,
                columns_to_filter=columns,
                filter_query=filter_query,
                interval=interval,
                use_pre_aggregates=True,
            )

        def build_raw_query():
            return table_schema._get_records_query(
                start=start,
                end=end,
                columns_to_filter=columns,
                filter_query=filter_query,
            )

        # Column mapping rules for pre-aggregate results (if needed)
        column_mapping_rules = {
            mm_schemas.WriterEvent.APPLICATION_NAME: [
                mm_schemas.WriterEvent.APPLICATION_NAME
            ],
            mm_schemas.MetricData.METRIC_NAME: [mm_schemas.MetricData.METRIC_NAME],
            mm_schemas.WriterEvent.ENDPOINT_ID: [mm_schemas.WriterEvent.ENDPOINT_ID],
        }

        df = self._connection.execute_with_fallback(
            self._pre_aggregate_manager,
            build_pre_agg_query,
            build_raw_query,
            interval=interval,
            agg_funcs=None,
            column_mapping_rules=column_mapping_rules,
            debug_name="get_metrics_metadata",
        )

        # Get distinct values
        if not df.empty:
            df = df.drop_duplicates()

        return df

    def calculate_latest_metrics(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        application_names: Optional[list[str]] = None,
    ) -> list[
        Union[mm_schemas.ApplicationResultRecord, mm_schemas.ApplicationMetricRecord]
    ]:
        """
        Calculate the latest metrics and results across applications.

        Returns a list of ApplicationResultRecord and ApplicationMetricRecord objects.

        :param start:              The start time of the query. Last 24 hours is used by default.
        :param end:                The end time of the query. The current time is used by default.
        :param application_names:  A list of application names to filter the results by. If not provided, all
                                applications are included.
        :return:                   A list containing the latest metrics and results for each application.
        """
        if not application_names:
            return []

        start, end = self._pre_aggregate_manager.get_start_end(start, end)

        metric_objects = []

        for app_name in application_names:
            # Get latest results for this application
            results_records = self._get_latest_results_for_application(
                app_name, start, end
            )
            metric_objects.extend(results_records)
            # Get latest metrics for this application
            metrics_records = self._get_latest_metrics_for_application(
                app_name, start, end
            )
            metric_objects.extend(metrics_records)
        return metric_objects

    def _get_latest_metrics_for_application(
        self, application_name: str, start: datetime, end: datetime
    ) -> list[mm_schemas.ApplicationMetricRecord]:
        """Get the latest metrics for a specific application."""
        table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]

        # Build filters using query builder utilities
        app_filter = TimescaleDBQueryBuilder.build_application_filter(application_name)
        time_filter = TimescaleDBQueryBuilder.build_time_range_filter(
            start, end, mm_schemas.WriterEvent.END_INFER_TIME
        )
        where_clause = TimescaleDBQueryBuilder.combine_filters(
            [app_filter, time_filter]
        )

        # DISTINCT ON is PostgreSQL-specific, keep as specialized query
        query = f"""
        SELECT DISTINCT ON (metric_name)
            {mm_schemas.WriterEvent.END_INFER_TIME},
            {mm_schemas.WriterEvent.APPLICATION_NAME},
            {mm_schemas.MetricData.METRIC_NAME},
            {mm_schemas.MetricData.METRIC_VALUE}
        FROM {table_schema.full_name()}
        WHERE {where_clause}
        ORDER BY metric_name, {mm_schemas.WriterEvent.END_INFER_TIME} DESC
        """

        stmt = Statement(query)
        result = self._connection.run(query=stmt)

        if not result or not result.data:
            return []

        # Work directly with raw result data instead of constructing DataFrame
        # Fields order: end_infer_time, application_name, metric_name, metric_value
        return [
            mm_schemas.ApplicationMetricRecord(
                time=row[0],  # end_infer_time
                value=row[3],  # metric_value
                metric_name=row[2],  # metric_name
            )
            for row in result.data
        ]

    def _get_latest_results_for_application(
        self, application_name: str, start: datetime, end: datetime
    ) -> list[mm_schemas.ApplicationResultRecord]:
        """Get the latest results for a specific application."""
        table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]

        # Build filters using query builder utilities
        app_filter = TimescaleDBQueryBuilder.build_application_filter(application_name)
        time_filter = TimescaleDBQueryBuilder.build_time_range_filter(
            start, end, mm_schemas.WriterEvent.END_INFER_TIME
        )
        where_clause = TimescaleDBQueryBuilder.combine_filters(
            [app_filter, time_filter]
        )

        # DISTINCT ON is PostgreSQL-specific, keep as specialized query
        query = f"""
        SELECT DISTINCT ON (result_name)
            {mm_schemas.WriterEvent.END_INFER_TIME},
            {mm_schemas.WriterEvent.APPLICATION_NAME},
            {mm_schemas.ResultData.RESULT_NAME},
            {mm_schemas.ResultData.RESULT_VALUE},
            {mm_schemas.ResultData.RESULT_STATUS},
            {mm_schemas.ResultData.RESULT_KIND}
        FROM {table_schema.full_name()}
        WHERE {where_clause}
        ORDER BY result_name, {mm_schemas.WriterEvent.END_INFER_TIME} DESC
        """

        stmt = Statement(query)
        result = self._connection.run(query=stmt)

        if not result or not result.data:
            return []

        # Work directly with raw result data instead of constructing DataFrame
        # Fields order: end_infer_time, application_name, result_name, result_value, result_status, result_kind
        return [
            mm_schemas.ApplicationResultRecord(
                time=row[0],  # end_infer_time
                value=row[3],  # result_value
                kind=mm_schemas.ResultKindApp(row[5]),  # result_kind
                status=mm_schemas.ResultStatusApp(row[4]),  # result_status
                result_name=row[2],  # result_name
            )
            for row in result.data
        ]
