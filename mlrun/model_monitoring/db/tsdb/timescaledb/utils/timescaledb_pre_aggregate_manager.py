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
from typing import Callable, Optional

import pandas as pd

import mlrun.common.schemas.model_monitoring as mm_schemas
from mlrun.model_monitoring.db.tsdb.preaggregate import PreAggregateHandler
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    TimescaleDBConnection,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_dataframe_processor import (
    TimescaleDBDataFrameProcessor,
)


class TimescaleDBPreAggregateManager:
    """
    Manages pre-aggregate queries with automatic fallback to raw data queries.

    This class encapsulates the common pattern of trying pre-aggregate queries first,
    then falling back to raw data queries if the pre-aggregate fails.
    """

    def __init__(
        self,
        pre_aggregate_handler: PreAggregateHandler,
        connection: TimescaleDBConnection,
    ):
        """
        Initialize the pre-aggregate manager.

        :param pre_aggregate_handler: Handler for pre-aggregate operations
        :param connection: TimescaleDB connection for executing queries
        """
        self.pre_aggregate_handler = pre_aggregate_handler
        self.connection = connection
        self.df_processor = TimescaleDBDataFrameProcessor()

    def execute_with_fallback(
        self,
        pre_agg_query_builder: Callable[[], str],
        raw_query_builder: Callable[[], str],
        interval: Optional[str] = None,
        agg_funcs: Optional[list[str]] = None,
        column_mapping_rules: Optional[dict[str, list[str]]] = None,
        debug_name: str = "query",
    ) -> pd.DataFrame:
        """
        Execute a query with pre-aggregate optimization and automatic fallback.

        :param pre_agg_query_builder: Function that returns pre-aggregate query string
        :param raw_query_builder: Function that returns raw data query string
        :param interval: Time interval for aggregation
        :param agg_funcs: List of aggregation functions
        :param column_mapping_rules: Rules for mapping column names in pre-aggregate results
        :param debug_name: Name for debugging/logging purposes
        :return: DataFrame with query results
        """
        if self.can_use_aggregates(interval, agg_funcs):
            try:
                # Try pre-aggregate query first
                query = pre_agg_query_builder()
                result = self.connection.run(query=query)
                df = self.df_processor.from_query_result(result)

                if not df.empty and column_mapping_rules:
                    # Apply flexible column mapping for pre-aggregate results
                    mapping = self.df_processor.build_flexible_column_mapping(
                        df, column_mapping_rules
                    )
                    df = self.df_processor.apply_column_mapping(df, mapping)

                return df

            except Exception as e:
                # Log the fallback (in production, use proper logging)
                print(
                    f"Pre-aggregate {debug_name} query failed, falling back to raw data: {e}"
                )

        # Fallback to raw data query
        raw_query = raw_query_builder()
        result = self.connection.run(query=raw_query)
        return self.df_processor.from_query_result(result)

    def can_use_aggregates(
        self, interval: Optional[str] = None, agg_funcs: Optional[list[str]] = None
    ) -> bool:
        """
        Check if pre-aggregates can be used for the given parameters.

        :param interval: Time interval for aggregation
        :param agg_funcs: List of aggregation functions
        :return: True if pre-aggregates can be used
        """
        return self.pre_aggregate_handler.can_use_pre_aggregates(
            interval=interval, agg_funcs=agg_funcs
        )

    def align_time_range(
        self, start: datetime, end: datetime, interval: Optional[str] = None
    ) -> tuple[datetime, datetime]:
        """
        Align time range to interval boundaries for optimal pre-aggregate usage.

        :param start: Start datetime
        :param end: End datetime
        :param interval: Time interval for alignment
        :return: Tuple of (aligned_start, aligned_end)
        """
        return self.pre_aggregate_handler.align_time_range(start, end, interval)

    def get_start_end(
        self, start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> tuple[datetime, datetime]:
        """
        Get normalized start and end times with defaults.

        :param start: Optional start datetime
        :param end: Optional end datetime
        :return: Tuple of (start, end) with defaults applied
        """
        return self.pre_aggregate_handler.get_start_end(start, end)

    def build_pre_aggregate_column_patterns(
        self, base_columns: list[str], agg_funcs: list[str]
    ) -> dict[str, list[str]]:
        """
        Build column name patterns for pre-aggregate results.

        Pre-aggregate queries often return columns with names like "avg_latency", "max_timestamp", etc.
        This method builds patterns to help map these back to expected column names.

        :param base_columns: List of base column names
        :param agg_funcs: List of aggregation functions used
        :return: Dictionary mapping target names to search patterns
        """
        patterns = {}

        for base_col in base_columns:
            base_name = base_col.split(".")[-1]  # Remove table prefixes
            search_patterns = [base_col, base_name]  # Start with exact matches

            # Add aggregated variations
            for func in agg_funcs:
                search_patterns.extend(
                    [
                        f"{func}_{base_name}",
                        f"{func}_{base_col}",
                        f"{func.upper()}_{base_name}",
                        f"{func.upper()}_{base_col}",
                    ]
                )

            patterns[base_name] = search_patterns

        return patterns

    def handle_endpoint_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize endpoint ID column naming across different query types.

        :param df: Input DataFrame
        :return: DataFrame with standardized endpoint column naming
        """
        endpoint_patterns = {
            "endpoint_id": [
                mm_schemas.WriterEvent.ENDPOINT_ID,
                "endpoint_id",
                "endpointid",
                "endpoint",
            ]
        }

        mapping = self.df_processor.build_flexible_column_mapping(df, endpoint_patterns)
        return self.df_processor.apply_column_mapping(df, mapping)

    def build_common_aggregation_query(
        self,
        table_schema,
        start: datetime,
        end: datetime,
        endpoint_filter: str,
        select_columns: list[str],
        group_by_columns: list[str],
        additional_filters: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> str:
        """
        Build a common aggregation query for raw data fallback.

        :param table_schema: Table schema object
        :param start: Start datetime
        :param end: End datetime
        :param endpoint_filter: Endpoint filter condition
        :param select_columns: Columns to select (including aggregation expressions)
        :param group_by_columns: Columns to group by
        :param additional_filters: Additional WHERE conditions
        :param order_by: ORDER BY clause
        :return: SQL query string
        """
        filters = [endpoint_filter]

        if additional_filters:
            filters.append(additional_filters)

        where_clause = " AND ".join(filters)

        query_parts = [
            f"SELECT {', '.join(select_columns)}",
            f"FROM {table_schema.full_name()}",
            f"WHERE {where_clause}",
            f"AND {table_schema.time_column} >= '{start}'",
            f"AND {table_schema.time_column} <= '{end}'",
        ]

        if group_by_columns:
            query_parts.append(f"GROUP BY {', '.join(group_by_columns)}")

        if order_by:
            query_parts.append(f"ORDER BY {order_by}")

        return "\n".join(query_parts) + ";"
