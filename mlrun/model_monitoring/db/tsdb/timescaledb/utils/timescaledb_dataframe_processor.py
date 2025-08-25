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

from typing import Optional

import pandas as pd

import mlrun.common.schemas.model_monitoring as mm_schemas
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    QueryResult,
)


class TimescaleDBDataFrameProcessor:
    """Utility class for common DataFrame processing operations."""

    @staticmethod
    def from_query_result(result: Optional[QueryResult]) -> pd.DataFrame:
        """
        Create a DataFrame from a QueryResult object.

        :param result: QueryResult object from TimescaleDB connection
        :return: pandas DataFrame
        """
        return pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

    @staticmethod
    def setup_time_index(
        df: pd.DataFrame, time_column: str, aggregation_window: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Set up time-based index for a DataFrame.

        :param df: Input DataFrame
        :param time_column: Name of the time column to use as index
        :param aggregation_window: Optional aggregation window that affects column naming
        :return: DataFrame with time index set up
        """
        if df.empty:
            return df

        # Determine the actual time column name based on aggregation
        if aggregation_window:
            # For pre-aggregated data, might use time_bucket
            actual_time_col = (
                "time_bucket" if "time_bucket" in df.columns else time_column
            )
        else:
            actual_time_col = time_column

        if actual_time_col in df.columns:
            df[actual_time_col] = pd.to_datetime(df[actual_time_col])
            df.set_index(actual_time_col, inplace=True)

        return df

    @staticmethod
    def apply_column_mapping(
        df: pd.DataFrame, mapping_config: dict[str, str]
    ) -> pd.DataFrame:
        """
        Apply column name mapping to a DataFrame.

        :param df: Input DataFrame
        :param mapping_config: Dictionary mapping old column names to new names
        :return: DataFrame with renamed columns
        """
        if df.empty or not mapping_config:
            return df

        if valid_mapping := {
            old: new for old, new in mapping_config.items() if old in df.columns
        }:
            df = df.rename(columns=valid_mapping, inplace=False)

        return df

    @staticmethod
    def handle_empty_dataframe(
        full_name: str, metric_type: str = "METRIC"
    ) -> mm_schemas.ModelEndpointMonitoringMetricNoData:
        """
        Create a standardized response for empty query results.

        :param full_name: Full metric name
        :param metric_type: Type of metric (METRIC or RESULT)
        :return: ModelEndpointMonitoringMetricNoData object
        """
        return mm_schemas.ModelEndpointMonitoringMetricNoData(
            full_name=full_name,
            type=getattr(
                mm_schemas.ModelEndpointMonitoringMetricType,
                metric_type,
                mm_schemas.ModelEndpointMonitoringMetricType.METRIC,
            ),
        )

    @staticmethod
    def convert_timestamp_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Convert specified columns to timestamps with proper error handling.

        :param df: Input DataFrame
        :param columns: List of column names to convert to timestamps
        :return: DataFrame with converted timestamp columns
        """
        if df.empty:
            return df

        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        return df

    @staticmethod
    def build_flexible_column_mapping(
        df: pd.DataFrame, target_patterns: dict[str, list[str]]
    ) -> dict[str, str]:
        """
        Build column mapping by finding columns that match target patterns.

        This handles cases where pre-aggregate queries return columns with different names
        than expected (e.g., "avg_latency" vs "latency").

        :param df: Input DataFrame
        :param target_patterns: Dict mapping target names to lists of patterns to search for
        :return: Dictionary mapping found column names to target names
        """
        if df.empty:
            return {}

        mapping = {}

        for target_name, patterns in target_patterns.items():
            found_col = next(
                (pattern for pattern in patterns if pattern in df.columns), None
            )
            # If no exact match, look for partial matches
            if not found_col:
                for col in df.columns:
                    col_lower = col.lower()
                    for pattern in patterns:
                        pattern_lower = pattern.lower()
                        if pattern_lower in col_lower or any(
                            word in col_lower for word in pattern_lower.split("_")
                        ):
                            found_col = col
                            break
                    if found_col:
                        break

            # Add to mapping if found and different from target
            if found_col and found_col != target_name:
                mapping[found_col] = target_name

        return mapping

    @staticmethod
    def ensure_required_columns(
        df: pd.DataFrame,
        required_columns: list[str],
        default_values: Optional[dict[str, any]] = None,
    ) -> pd.DataFrame:
        """
        Ensure DataFrame has all required columns, adding missing ones with default values.

        :param df: Input DataFrame
        :param required_columns: List of column names that must exist
        :param default_values: Optional dict of default values for missing columns
        :return: DataFrame with all required columns
        """
        if df.empty:
            return df

        default_values = default_values or {}

        for col in required_columns:
            if col not in df.columns:
                df[col] = default_values.get(col, None)

        return df
