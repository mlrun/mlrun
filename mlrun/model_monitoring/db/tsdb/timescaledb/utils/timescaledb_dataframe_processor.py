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
            df = df.rename(columns=valid_mapping)

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
    def build_flexible_column_mapping(
        df: pd.DataFrame, target_patterns: dict[str, list[str]]
    ) -> dict[str, str]:
        """
        Build column mapping by finding columns that match target patterns.

        This handles cases where pre-aggregate queries return columns with different names
        than expected (e.g., "avg_latency" vs "latency").

        LIMITATION: This method uses first-match logic and cannot handle multiple aggregates
        of the same base column (e.g., both "avg_latency" and "max_latency" in the same DataFrame).
        It's designed for single-aggregate scenarios or pre-aggregate fallback mapping.

        For multi-aggregate support, each aggregate should have distinct target patterns.

        :param df: Input DataFrame
        :param target_patterns: Dict mapping target names to lists of patterns to search for.
                            Each target should map to only one expected column in the DataFrame.
        :return: Dictionary mapping found column names to target names
        """
        if df.empty:
            return {}

        # Pre-compute all patterns once
        exact_patterns = {}  # pattern -> target_name (case-insensitive)
        fuzzy_patterns = []  # (target_name, pattern_words_set) for O(1) word lookups

        for target_name, patterns in target_patterns.items():
            for pattern in patterns:
                pattern_lower = pattern.lower()
                exact_patterns[pattern_lower] = target_name

                # Use set for O(1) word lookup instead of list
                pattern_words = set(pattern_lower.split("_"))
                fuzzy_patterns.append((target_name, pattern_words))

        mapping = {}

        # Process columns with early termination
        for col in df.columns:
            col_lower = col.lower()

            # Exact match - O(1)
            if target_name := exact_patterns.get(col_lower):
                if col != target_name:  # Only add if different
                    mapping[col] = target_name
                continue

            # Fuzzy match - optimized with set operations
            col_words = set(col_lower.split("_"))

            for target_name, pattern_words in fuzzy_patterns:
                # Require ALL pattern words to be present (subset match)
                # This ensures "avg_latency" pattern only matches columns containing both words
                if pattern_words.issubset(col_words):
                    if col != target_name:
                        mapping[col] = target_name
                    break

        return mapping
