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

import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, Union

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors

if TYPE_CHECKING:
    import pandas as pd

# TimescaleDB interval pattern for parsing intervals like "1h", "10m", "1d", "1w", "1M"
_TIMESCALEDB_INTERVAL_PATTERN = re.compile(r"(\d+)([mhdwM])")


class TimescaleDBQueryBuilder:
    """Utility class for building common SQL query components."""

    @staticmethod
    def build_endpoint_filter(endpoint_ids: Optional[Union[str, list[str]]]) -> str:
        """
        Generate SQL filter for endpoint IDs.

        :param endpoint_ids: Single endpoint ID, list of endpoint IDs, or None for no filtering
        :return: SQL WHERE clause fragment for endpoint filtering, or empty string if None
        """
        if endpoint_ids is None:
            return ""
        if isinstance(endpoint_ids, str):
            return f"{mm_schemas.WriterEvent.ENDPOINT_ID}='{endpoint_ids}'"
        elif isinstance(endpoint_ids, list):
            endpoint_list = "', '".join(endpoint_ids)
            return f"{mm_schemas.WriterEvent.ENDPOINT_ID} IN ('{endpoint_list}')"
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid 'endpoint_ids' filter: must be a string or a list of strings"
            )

    @staticmethod
    def build_time_range_filter(
        start: datetime, end: datetime, time_column: str
    ) -> str:
        """
        Generate SQL filter for time range.

        :param start: Start datetime
        :param end: End datetime
        :param time_column: Name of the time column to filter on
        :return: SQL WHERE clause fragment for time filtering
        """
        return f"{time_column} >= '{start}' AND {time_column} <= '{end}'"

    @staticmethod
    def build_application_filter(app_names: Union[str, list[str]]) -> str:
        """
        Generate SQL filter for application names.

        :param app_names: Single application name or list of application names
        :return: SQL WHERE clause fragment for application filtering
        """
        if isinstance(app_names, str):
            return f"{mm_schemas.WriterEvent.APPLICATION_NAME} = '{app_names}'"
        elif isinstance(app_names, list):
            app_list = "', '".join(app_names)
            return f"{mm_schemas.WriterEvent.APPLICATION_NAME} IN ('{app_list}')"
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid 'app_names' filter: must be either a string or a list of strings"
            )

    @staticmethod
    def build_metrics_filter(
        metrics: Optional[list[mm_schemas.ModelEndpointMonitoringMetric]],
    ) -> str:
        """
        Generate SQL filter for metrics using both application_name and metric_name columns.

        :param metrics: List of ModelEndpointMonitoringMetric objects, or None for no filtering
        :return: SQL WHERE clause fragment for metrics filtering, or empty string if None
        """
        if metrics is None:
            return ""
        if not metrics:
            raise mlrun.errors.MLRunInvalidArgumentError("Metrics list cannot be empty")

        # Build filter that includes both application_name and metric_name
        # Format: (application_name = 'app1' AND metric_name = 'name1') OR
        # (application_name = 'app2' AND metric_name = 'name2')
        conditions = []
        for metric in metrics:
            condition = (
                f"({mm_schemas.WriterEvent.APPLICATION_NAME} = '{metric.app}' "
                f"AND {mm_schemas.MetricData.METRIC_NAME} = '{metric.name}')"
            )
            conditions.append(condition)

        if len(conditions) == 1:
            return conditions[0]
        return " OR ".join(conditions)

    @staticmethod
    def build_results_filter(
        metrics: Optional[list[mm_schemas.ModelEndpointMonitoringMetric]],
    ) -> str:
        """
        Generate SQL filter for results using both application_name and result_name columns.
        :param metrics: List of ModelEndpointMonitoringMetric objects, or None for no filtering
        :return: SQL WHERE clause fragment for results filtering, or empty string if None
        """
        if metrics is None:
            return ""
        if not metrics:
            raise mlrun.errors.MLRunInvalidArgumentError("Metrics list cannot be empty")

        # Build filter that includes both application_name and result_name
        # Format: (application_name = 'app1' AND result_name = 'name1') OR
        # (application_name = 'app2' AND result_name = 'name2')
        conditions = []
        for metric in metrics:
            condition = (
                f"({mm_schemas.WriterEvent.APPLICATION_NAME} = '{metric.app}' "
                f"AND {mm_schemas.ResultData.RESULT_NAME} = '{metric.name}')"
            )
            conditions.append(condition)

        if len(conditions) == 1:
            return conditions[0]
        return " OR ".join(conditions)

    @staticmethod
    def build_metrics_filter_from_names(metric_names: list[str]) -> str:
        """
        Generate SQL filter for metrics by name.

        :param metric_names: List of metric names
        :return: SQL WHERE clause fragment for metrics filtering
        """
        if not metric_names:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Metric names list cannot be empty"
            )

        if len(metric_names) == 1:
            return f"{mm_schemas.MetricData.METRIC_NAME} = '{metric_names[0]}'"
        metric_list = "', '".join(metric_names)
        return f"{mm_schemas.MetricData.METRIC_NAME} IN ('{metric_list}')"

    @staticmethod
    def combine_filters(filters: list[str]) -> Optional[str]:
        """
        Combine multiple filter conditions with AND operator.

        :param filters: List of filter condition strings
        :return: Combined filter string or None if no filters
        """
        if valid_filters := [f.strip() for f in filters if f.strip()]:
            return (
                valid_filters[0]
                if len(valid_filters) == 1
                else " AND ".join(valid_filters)
            )
        else:
            return None

    @staticmethod
    def interval_to_minutes(interval: str) -> Optional[int]:
        """
        Convert TimescaleDB interval string to minutes.

        Uses PostgreSQL/TimescaleDB fixed duration assumptions:
        - 1 month = 30 days = 43,200 minutes
        - 1 year = 365.25 days = 525,960 minutes

        This matches TimescaleDB's INTERVAL arithmetic behavior and is appropriate
        for duration calculations and optimal interval selection.

        :param interval: Interval string like "1h", "10m", "1d", "1w", "1M"
        :return: Duration in minutes, or None if invalid format
        """
        match = _TIMESCALEDB_INTERVAL_PATTERN.match(interval)
        if not match:
            return None

        amount, unit = int(match.group(1)), match.group(2)

        if unit == "m":  # minutes
            return amount
        elif unit == "h":  # hours
            return amount * 60
        elif unit == "d":  # days
            return amount * 1440
        elif unit == "w":  # weeks
            return amount * 10080
        elif unit == "M":  # months (PostgreSQL: 30 days)
            return amount * 43200
        else:
            return None

    @staticmethod
    def determine_optimal_interval(start: datetime, end: datetime) -> str:
        """
        Determine optimal interval for time-based aggregation based on time range.

        This method selects appropriate interval from a comprehensive list of
        standard TimescaleDB intervals rather than simple time-based thresholds.
        This provides better balance between query performance
        and data granularity by targeting optimal data point counts.

        :param start: Start time
        :param end: End time
        :return: Optimal interval string (in Python format like "1h", "1d")
        """
        # Comprehensive list of standard TimescaleDB intervals
        standard_intervals = [
            "1m",
            "5m",
            "10m",
            "15m",
            "30m",
            "1h",
            "2h",
            "6h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        ]

        optimal = TimescaleDBQueryBuilder.determine_optimal_from_available(
            start, end, standard_intervals
        )

        # Fallback for edge cases where algorithm doesn't find a suitable match
        # Simple binary choice: smallest interval for short ranges, largest for long ranges
        if optimal is None:
            time_diff = end - start
            return "1m" if time_diff <= timedelta(days=30) else "1M"
        return optimal

    @staticmethod
    def determine_optimal_from_available(
        start: datetime, end: datetime, available_intervals: list[str]
    ) -> Optional[str]:
        """
        Determine optimal interval from available pre-aggregate intervals.

        Uses a formula-based approach to select intervals that provide reasonable data points
        (~50-200 range) for optimal visualization and query performance.

        :param start: Start time
        :param end: End time
        :param available_intervals: List of available interval strings (e.g., ["10m", "1h", "6h", "1d"])
        :return: Optimal interval string or None if no suitable intervals available
        """
        if not available_intervals:
            return None

        # Convert available intervals to (name, minutes) tuples using our centralized parsing
        available_with_minutes = []
        for interval in available_intervals:
            minutes = TimescaleDBQueryBuilder.interval_to_minutes(interval)
            if minutes is not None:
                available_with_minutes.append((interval, minutes))

        if not available_with_minutes:
            return None

        # Sort by duration (ascending)
        available_with_minutes.sort(key=lambda x: x[1])

        # Calculate time range in minutes
        time_diff_minutes = (end - start).total_seconds() / 60

        # Target ~100 data points for optimal visualization balance
        # Accept intervals that give 20-500 data points (wider reasonable range)
        target_points = 100
        min_acceptable_points = 20
        max_acceptable_points = 500

        optimal_interval_minutes = time_diff_minutes / target_points
        min_interval_minutes = time_diff_minutes / max_acceptable_points
        max_interval_minutes = time_diff_minutes / min_acceptable_points

        # Find the best matching interval within acceptable range
        best_interval = None
        best_score = float("inf")

        for interval_name, interval_minutes in available_with_minutes:
            # Check if this interval is within acceptable range
            if min_interval_minutes <= interval_minutes <= max_interval_minutes:
                # Score by distance from optimal (closer to optimal = better)
                score = abs(interval_minutes - optimal_interval_minutes)
                if score < best_score:
                    best_score = score
                    best_interval = interval_name

        return best_interval

    @staticmethod
    def build_read_data_with_fallback(
        connection,
        pre_aggregate_manager,
        table_schema,
        start: "datetime",  # Use string to avoid import cycle
        end: "datetime",
        columns: list[str],
        filter_query: Optional[str],
        name_column: str,
        value_column: str,
        debug_name: str = "read_data",
    ) -> "pd.DataFrame":  # Use string to avoid import cycle
        """
        Build and execute read data query with pre-aggregate fallback pattern.

        This method deduplicates the common pattern used in both metrics and results
        queries for reading data with pre-aggregate optimization and fallback.

        :param connection: Database connection instance
        :param pre_aggregate_manager: Pre-aggregate handler for optimization
        :param table_schema: Table schema for query building
        :param start: Start datetime for query
        :param end: End datetime for query
        :param columns: List of columns to select
        :param filter_query: WHERE clause conditions
        :param name_column: Name of the metric/result name column
        :param value_column: Name of the metric/result value column
        :param debug_name: Name for debugging purposes
        :return: DataFrame with query results
        """

        def build_pre_agg_query():
            return table_schema._get_records_query(
                start=start,
                end=end,
                columns_to_filter=columns,
                filter_query=filter_query,
                use_pre_aggregates=True,
            )

        def build_raw_query():
            return table_schema._get_records_query(
                start=start,
                end=end,
                columns_to_filter=columns,
                filter_query=filter_query,
            )

        # Column mapping rules for pre-aggregate results
        import mlrun.common.schemas.model_monitoring as mm_schemas

        column_mapping_rules = {
            name_column: [name_column],
            value_column: [value_column],
            table_schema.time_column: [table_schema.time_column],
            mm_schemas.WriterEvent.APPLICATION_NAME: [
                mm_schemas.WriterEvent.APPLICATION_NAME
            ],
        }

        return connection.execute_with_fallback(
            pre_aggregate_manager,
            build_pre_agg_query,
            build_raw_query,
            interval=None,  # No specific interval for this query
            agg_funcs=None,
            column_mapping_rules=column_mapping_rules,
            debug_name=debug_name,
        )

    @staticmethod
    def prepare_time_range_and_interval(
        pre_aggregate_manager,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
        auto_determine_interval: bool = True,
    ) -> tuple[datetime, datetime, str]:
        """
        Standardized time range and interval preparation for TimescaleDB queries.

        This helper eliminates the common pattern of:
        1. get_start_end()
        2. determine_optimal_interval() (optional)
        3. align_time_range()

        :param pre_aggregate_manager: PreAggregateManager instance
        :param start: Start datetime (optional)
        :param end: End datetime (optional)
        :param interval: Time interval (optional, auto-determined if None and auto_determine_interval=True)
        :param auto_determine_interval: Whether to auto-determine interval if not provided
        :return: Tuple of (aligned_start, aligned_end, interval) - interval is guaranteed to be valid
        """
        # Step 1: Get start/end times with defaults
        start, end = pre_aggregate_manager.get_start_end(start, end)

        # Step 2: Auto-determine optimal interval if requested and not provided
        if interval is None and auto_determine_interval:
            # First, try to use available pre-aggregate intervals if they exist
            available_intervals = (
                pre_aggregate_manager.config.aggregate_intervals
                if pre_aggregate_manager.config
                else None
            )

            if available_intervals:
                if optimal_from_preaggregate := (
                    TimescaleDBQueryBuilder.determine_optimal_from_available(
                        start, end, available_intervals
                    )
                ):
                    interval = optimal_from_preaggregate

            # If no suitable pre-aggregate interval found, use formula-based approach
            if interval is None:
                interval = TimescaleDBQueryBuilder.determine_optimal_interval(
                    start, end
                )

        # Step 3: Align times to interval boundaries
        start, end = pre_aggregate_manager.align_time_range(start, end, interval)

        return start, end, interval

    @staticmethod
    def prepare_time_range_with_validation(
        pre_aggregate_manager,
        start_iso: str,
        end_iso: str,
        interval: Optional[str] = None,
        agg_function: Optional[str] = None,
    ) -> tuple[datetime, datetime, Optional[str]]:
        """
        Specialized helper for time preparation with validation and ISO string conversion.

        This helper eliminates the pattern of:
        1. validate_interval_and_function()
        2. datetime.fromisoformat() conversion
        3. align_time_range()

        :param pre_aggregate_manager: PreAggregateManager instance
        :param start_iso: Start time as ISO string
        :param end_iso: End time as ISO string
        :param interval: Time interval (optional)
        :param agg_function: Aggregation function (optional)
        :return: Tuple of (aligned_start_dt, aligned_end_dt, interval)
        """
        # Step 1: Validate parameters using the pre-aggregate handler
        pre_aggregate_manager.validate_interval_and_function(interval, agg_function)

        # Step 2: Convert ISO strings to datetime objects
        start_dt, end_dt = (
            datetime.fromisoformat(start_iso),
            datetime.fromisoformat(end_iso),
        )

        # Step 3: Align times if interval is provided
        start_dt, end_dt = pre_aggregate_manager.align_time_range(
            start_dt, end_dt, interval
        )

        return start_dt, end_dt, interval

    @staticmethod
    def build_endpoint_aggregation_query(
        subquery: str,
        aggregation_columns: dict[str, str],
        group_by_column: str = mm_schemas.WriterEvent.ENDPOINT_ID,
        order_by_column: str = mm_schemas.WriterEvent.ENDPOINT_ID,
    ) -> str:
        """
        Build standardized outer query for endpoint-level aggregation over time buckets.

        This helper eliminates the repeated pattern of:
        SELECT endpoint_id, AGG(column) FROM (subquery) GROUP BY endpoint_id ORDER BY endpoint_id

        :param subquery: Inner query that provides time-bucketed data
        :param aggregation_columns: Dict of {result_column: "AGG(source_column)"} mappings
        :param group_by_column: Column to group by (default: endpoint_id)
        :param order_by_column: Column to order by (default: endpoint_id)
        :return: Complete SQL query string
        """
        # Build the SELECT columns list
        select_columns = [group_by_column] + [
            f"{agg_expr} AS {result_col}"
            for result_col, agg_expr in aggregation_columns.items()
        ]

        return f"""
        SELECT
            {', '.join(select_columns)}
        FROM ({subquery}) AS time_buckets
        GROUP BY {group_by_column}
        ORDER BY {order_by_column}
        """


class TimescaleDBNaming:
    """Utility class for TimescaleDB table and view naming conventions."""

    @staticmethod
    def get_agg_table_name(base_name: str, interval: str) -> str:
        """
        Generate aggregate table name with interval.

        :param base_name: Base table name
        :param interval: Time interval (e.g., '1h', '1d')
        :return: Aggregate table name (e.g., 'metrics_agg_1h')
        """
        return f"{base_name}_agg_{interval}"

    @staticmethod
    def get_cagg_view_name(base_name: str, interval: str) -> str:
        """
        Generate continuous aggregate view name with interval.

        :param base_name: Base table name
        :param interval: Time interval (e.g., '1h', '1d')
        :return: Continuous aggregate view name (e.g., 'metrics_cagg_1h')
        """
        return f"{base_name}_cagg_{interval}"

    @staticmethod
    def get_agg_pattern(base_pattern: str) -> str:
        """
        Generate SQL LIKE pattern for aggregate tables.

        :param base_pattern: Base pattern (e.g., 'metrics')
        :return: SQL LIKE pattern (e.g., 'metrics_agg_%')
        """
        return f"{base_pattern}_agg_%"

    @staticmethod
    def get_cagg_pattern(base_pattern: str) -> str:
        """
        Generate SQL LIKE pattern for continuous aggregate views.

        :param base_pattern: Base pattern (e.g., 'metrics')
        :return: SQL LIKE pattern (e.g., 'metrics_cagg_%')
        """
        return f"{base_pattern}_cagg_%"

    @staticmethod
    def get_all_aggregate_patterns(base_pattern: str) -> list[str]:
        """
        Generate both aggregate table and continuous aggregate view patterns.

        :param base_pattern: Base pattern (e.g., 'metrics')
        :return: List of patterns ['metrics_agg_%', 'metrics_cagg_%']
        """
        return [
            TimescaleDBNaming.get_agg_pattern(base_pattern),
            TimescaleDBNaming.get_cagg_pattern(base_pattern),
        ]

    @staticmethod
    def get_deletion_patterns(base_pattern: str) -> list[str]:
        """
        Generate all patterns needed for table deletion operations.

        :param base_pattern: Base pattern (e.g., 'metrics')
        :return: List of patterns [base_pattern, 'metrics_agg_%', 'metrics_cagg_%']
        """
        return [
            base_pattern,
            TimescaleDBNaming.get_agg_pattern(base_pattern),
            TimescaleDBNaming.get_cagg_pattern(base_pattern),
        ]
