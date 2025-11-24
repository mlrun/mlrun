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

import pandas as pd

from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_dataframe_processor import (
    TimescaleDBDataFrameProcessor,
)


class TestTimescaleDBDataFrameProcessor:
    """Test suite for TimescaleDBDataFrameProcessor."""

    def test_build_flexible_column_mapping_successful_cases(self):
        """Test successful column mapping scenarios: exact match, case insensitive, and fuzzy matching."""
        processor = TimescaleDBDataFrameProcessor()

        # Test exact match
        df_exact = pd.DataFrame(
            {"avg_latency": [1, 2, 3], "endpoint_id": ["a", "b", "c"]}
        )
        target_patterns = {
            "latency": ["avg_latency", "latency"],
            "endpoint": ["endpoint_id", "endpoint"],
        }
        mapping = processor.build_flexible_column_mapping(df_exact, target_patterns)
        expected = {"avg_latency": "latency", "endpoint_id": "endpoint"}
        assert mapping == expected

        # Test case insensitive matching
        df_case = pd.DataFrame(
            {"AVG_LATENCY": [1, 2, 3], "Endpoint_ID": ["a", "b", "c"]}
        )
        mapping = processor.build_flexible_column_mapping(df_case, target_patterns)
        expected = {"AVG_LATENCY": "latency", "Endpoint_ID": "endpoint"}
        assert mapping == expected

        # Test fuzzy matching using word intersection
        df_fuzzy = pd.DataFrame(
            {
                "time_bucket_avg_latency": [1, 2, 3],  # Contains 'avg' and 'latency'
                "custom_endpoint_data": ["a", "b", "c"],  # Contains 'endpoint'
            }
        )
        mapping = processor.build_flexible_column_mapping(df_fuzzy, target_patterns)
        expected = {
            "time_bucket_avg_latency": "latency",
            "custom_endpoint_data": "endpoint",
        }
        assert mapping == expected

    def test_build_flexible_column_mapping_edge_cases(self):
        """Test edge cases: empty DataFrame, no matches, and identity columns."""
        processor = TimescaleDBDataFrameProcessor()
        target_patterns = {
            "latency": ["avg_latency", "latency"],
            "endpoint": ["endpoint_id", "endpoint"],
        }

        # Test empty DataFrame
        df_empty = pd.DataFrame()
        mapping = processor.build_flexible_column_mapping(df_empty, target_patterns)
        assert mapping == {}

        # Test no matches
        df_no_match = pd.DataFrame(
            {"random_column": [1, 2, 3], "another_col": ["a", "b", "c"]}
        )
        mapping = processor.build_flexible_column_mapping(df_no_match, target_patterns)
        assert mapping == {}

        # Test identity columns (same name as target)
        df_identity = pd.DataFrame({"latency": [1, 2, 3], "endpoint": ["a", "b", "c"]})
        mapping = processor.build_flexible_column_mapping(df_identity, target_patterns)
        assert mapping == {}  # No mapping needed when names already match

    def test_build_flexible_column_mapping_complex_scenarios(self):
        """Test complex realistic scenarios with mixed matching patterns and priorities."""
        processor = TimescaleDBDataFrameProcessor()

        # Test mixed scenario with multiple match types
        df_mixed = pd.DataFrame(
            {
                "time_bucket": [1, 2, 3],  # No match expected
                "avg_latency": [10, 20, 30],  # Exact match -> latency
                "endpoint_id": ["a", "b", "c"],  # Exact match -> endpoint
                "custom_metric_data": [100, 200, 300],  # Fuzzy match -> metric
                "random_field": ["x", "y", "z"],  # No match
            }
        )
        target_patterns = {
            "latency": ["avg_latency", "latency"],
            "endpoint": ["endpoint_id", "endpoint"],
            "metric": ["metric_value", "metric"],
        }
        mapping = processor.build_flexible_column_mapping(df_mixed, target_patterns)
        expected = {
            "avg_latency": "latency",
            "endpoint_id": "endpoint",
            "custom_metric_data": "metric",
        }
        assert mapping == expected

        # Test multiple patterns per target and priority handling
        df_priority = pd.DataFrame(
            {
                "average_latency": [1, 2, 3],  # Should match latency patterns
                "avg_latency_metric": [
                    10,
                    20,
                    30,
                ],  # Could match both latency and metric
            }
        )
        patterns_priority = {
            "latency": ["avg_latency", "average_latency", "latency", "request_latency"],
            "metric": ["latency_metric", "metric"],
        }
        mapping = processor.build_flexible_column_mapping(
            df_priority, patterns_priority
        )

        # Verify both columns are mapped and priority is respected
        assert "average_latency" in mapping
        assert mapping["average_latency"] == "latency"
        assert "avg_latency_metric" in mapping
        assert mapping["avg_latency_metric"] in [
            "latency",
            "metric",
        ]  # Either is acceptable
