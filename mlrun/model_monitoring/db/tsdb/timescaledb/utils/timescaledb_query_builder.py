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

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors


class TimescaleDBQueryBuilder:
    """Utility class for building common SQL query components."""

    @staticmethod
    def build_endpoint_filter(endpoint_ids: Union[str, list[str]]) -> str:
        """
        Generate SQL filter for endpoint IDs.

        :param endpoint_ids: Single endpoint ID or list of endpoint IDs
        :return: SQL WHERE clause fragment for endpoint filtering
        """
        if isinstance(endpoint_ids, str):
            return f"{mm_schemas.WriterEvent.ENDPOINT_ID}='{endpoint_ids}'"
        elif isinstance(endpoint_ids, list):
            endpoint_list = "', '".join(endpoint_ids)
            return f"{mm_schemas.WriterEvent.ENDPOINT_ID} IN ('{endpoint_list}')"
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid 'endpoint_ids' filter: must be a string or a list."
            )

    @staticmethod
    def build_time_range_filter(start, end, time_column: str) -> str:
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
                "Invalid 'app_names' filter: must be a string or a list."
            )

    @staticmethod
    def build_metrics_condition(
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric],
    ) -> str:
        """
        Build OR condition for metrics filtering.

        :param metrics: List of metric objects with app and name attributes
        :return: SQL WHERE clause fragment for metrics filtering
        """
        metrics_conditions = []
        for metric in metrics:
            condition = (
                f"({mm_schemas.WriterEvent.APPLICATION_NAME}='{metric.app}' "
                f"AND {mm_schemas.MetricData.METRIC_NAME}='{metric.name}')"
            )
            metrics_conditions.append(condition)
        return " OR ".join(metrics_conditions)

    @staticmethod
    def build_parameterized_query(base_query: str, params: list) -> str:
        """
        Build parameterized query with proper placeholder handling.

        :param base_query: Base SQL query with %s placeholders
        :param params: List of parameters to substitute
        :return: Query string with parameters substituted (for logging/debugging)
        """
        # This is mainly for debugging - actual execution should use Statement objects
        try:
            return base_query % tuple(params)
        except (TypeError, ValueError):
            return f"{base_query} [PARAMS: {params}]"

    @staticmethod
    def build_aggregation_columns(
        agg_funcs: list[str], base_columns: list[str]
    ) -> list[str]:
        """
        Build column names for aggregation queries.

        :param agg_funcs: List of aggregation functions (e.g., ['avg', 'max'])
        :param base_columns: List of base column names to aggregate
        :return: List of aggregated column expressions
        """
        agg_columns = []
        for func in agg_funcs:
            agg_columns.extend(
                f"{func.upper()}({col}) as {func}_{col}" for col in base_columns
            )
        return agg_columns

    @staticmethod
    def combine_filters(filters: list[str], operator: str = "AND") -> Optional[str]:
        """
        Combine multiple filter conditions with the specified operator.

        :param filters: List of filter condition strings
        :param operator: SQL operator to use (AND/OR)
        :return: Combined filter string or None if no filters
        """
        if valid_filters := [f for f in filters if f and f.strip()]:
            return (
                valid_filters[0]
                if len(valid_filters) == 1
                else f" {operator} ".join(valid_filters)
            )
        else:
            return None

    @staticmethod
    def determine_optimal_interval(start: datetime, end: datetime) -> str:
        """
        Determine optimal interval for time-based aggregation based on time range.

        This method selects appropriate intervals to balance query performance
        and data granularity based on the total time span.

        :param start: Start time
        :param end: End time
        :return: Optimal interval string (in Python format like "1h", "1d")
        """
        from datetime import timedelta

        # Calculate the time difference to determine appropriate interval
        time_diff = end - start

        if time_diff <= timedelta(hours=6):
            # For short periods, use 1 hour intervals
            return "1h"
        elif time_diff <= timedelta(days=2):
            # For medium periods, use 1 hour intervals
            return "1h"
        elif time_diff <= timedelta(days=7):
            # For week-long periods, use 6 hour intervals
            return "6h"
        else:
            # For longer periods, use daily intervals
            return "1d"

    @staticmethod
    def parse_datetime_strings(start: str, end: str) -> tuple[datetime, datetime]:
        """
        Parse ISO format datetime strings to datetime objects.

        :param start: Start datetime in ISO format string
        :param end: End datetime in ISO format string
        :return: Tuple of (start_datetime, end_datetime)
        """
        return datetime.fromisoformat(start), datetime.fromisoformat(end)


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
