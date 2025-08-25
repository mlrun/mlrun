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

import os
from datetime import datetime, timedelta

import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_query_builder import (
    TimescaleDBQueryBuilder,
)

# Skip entire module if connection string is not available or not PostgreSQL
connection_string = os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__TSDB_CONNECTION")
pytestmark = pytest.mark.skipif(
    not connection_string or not connection_string.startswith("postgres"),
    reason="TimescaleDB connection string not available or not PostgreSQL",
)


class TestParameterValidation:
    """Tests for parameter validation and error handling."""

    def test_read_predictions_invalid_aggregation_params(self, query_handler):
        """Test MLRunInvalidArgumentError when agg_funcs without aggregation_window."""
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
            query_handler.read_predictions(
                endpoint_id="test_endpoint",
                start=start_time,
                end=end_time,
                agg_funcs=["avg"],  # agg_funcs provided but no aggregation_window
            )

        assert (
            "both or neither of `aggregation_window` and `agg_funcs` must be provided"
            in str(exc_info.value)
        )

    def test_read_predictions_invalid_aggregation_params_reverse(self, query_handler):
        """Test MLRunInvalidArgumentError when aggregation_window without agg_funcs."""
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
            query_handler.read_predictions(
                endpoint_id="test_endpoint",
                start=start_time,
                end=end_time,
                aggregation_window="1h",  # aggregation_window provided but no agg_funcs
            )

        assert (
            "both or neither of `aggregation_window` and `agg_funcs` must be provided"
            in str(exc_info.value)
        )

    def test_get_endpoint_filter_invalid_type(self, query_handler):
        """Test MLRunInvalidArgumentError for invalid endpoint_id type."""
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
            TimescaleDBQueryBuilder.build_endpoint_filter(
                123
            )  # Invalid type - should be string or list

        assert "Invalid 'endpoint_ids' filter: must be a string or a list" in str(
            exc_info.value
        )

    def test_get_endpoint_filter_dict_type(self, query_handler):
        """Test MLRunInvalidArgumentError for dict endpoint_id type."""
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
            TimescaleDBQueryBuilder.build_endpoint_filter(
                {"endpoint": "test"}
            )  # Invalid type - should be string or list

        assert "Invalid 'endpoint_ids' filter: must be a string or a list" in str(
            exc_info.value
        )


class TestPreAggregateExceptionHandling:
    """Tests for pre-aggregate exception handling and fallback logic."""

    # COMMENTED OUT: Redundant fallback tests - user requirement is "No fallback! Pre-aggregate must work"
    # Better pre-aggregate tests exist in test_timescaledb_queries_aggregation.py with working pre-aggregates
    # def test_get_avg_latency_pre_aggregate_exception_fallback(self, query_handler_with_aggregates):
    #     """Test fallback to raw data when pre-aggregate query fails."""
    #     # Mock the _connection.run to raise exception on pre-aggregate query
    #     original_run = query_handler_with_aggregates._connection.run
    #     call_count = 0

    #     def mock_run(query=None, statements=None):
    #         nonlocal call_count
    #         call_count += 1

    #         # First call (pre-aggregate) should fail, second call (raw data) should succeed
    #         if call_count == 1 and "use_pre_aggregates=True" in str(query) if query else False:
    #             raise Exception("Pre-aggregate query failed")
    #         else:
    #             return original_run(query=query, statements=statements)

    #     with patch.object(query_handler_with_aggregates._connection, 'run', side_effect=mock_run):
    #         # This should trigger the exception handling path
    #         result = query_handler_with_aggregates.get_avg_latency(
    #             endpoint_ids=["test_endpoint"],
    #             interval="1h"
    #         )

    #     # Should have fallen back to raw data and returned empty DataFrame
    #     assert isinstance(result, pd.DataFrame)
    #     assert len(result) == 0  # No data inserted, so empty result expected

    # def test_get_drift_status_pre_aggregate_exception_fallback(self, query_handler_with_aggregates):
    #     """Test fallback to raw data when pre-aggregate query fails in get_drift_status."""
    #     # Mock the _connection.run to raise exception on pre-aggregate query
    #     original_run = query_handler_with_aggregates._connection.run
    #     call_count = 0

    #     def mock_run(query=None, statements=None):
    #         nonlocal call_count
    #         call_count += 1

    #         # First call (pre-aggregate) should fail, second call (raw data) should succeed
    #         if call_count == 1 and "use_pre_aggregates=True" in str(query) if query else False:
    #             raise Exception("Pre-aggregate query failed")
    #         else:
    #             return original_run(query=query, statements=statements)

    #     with patch.object(query_handler_with_aggregates._connection, 'run', side_effect=mock_run):
    #         # This should trigger the exception handling path
    #         result = query_handler_with_aggregates.get_drift_status(
    #             endpoint_ids=["test_endpoint"],
    #             interval="1h"
    #         )

    #     # Should have fallen back to raw data and returned empty DataFrame
    #     assert isinstance(result, pd.DataFrame)
    #     assert len(result) == 0  # No data inserted, so empty result expected

    def test_calculate_latest_metrics_with_error_data_only(self, query_handler):
        """Test calculate_latest_metrics with only error data (no metrics or results)."""
        # Insert error data only - this tests edge case handling
        connection = query_handler._connection
        errors_table = query_handler.tables[mm_schemas.TimescaleDBTables.ERRORS]

        test_time = datetime(2024, 1, 15, 12, 0, 0)
        connection.run(
            statements=[
                f"""
                INSERT INTO {errors_table.full_name()}
                ({mm_schemas.EventFieldType.TIME}, {mm_schemas.WriterEvent.ENDPOINT_ID},
                 {mm_schemas.EventFieldType.MODEL_ERROR}, {mm_schemas.EventFieldType.ERROR_TYPE})
                VALUES ('{test_time}', 'test_endpoint', 'Test error message', 'inference_error')
                """
            ]
        )

        result = query_handler.calculate_latest_metrics(
            application_names=["test_app"],
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
        )

        # Should return empty list since errors are not included in latest metrics calculation
        assert isinstance(result, list)
        assert len(result) == 0

    def test_calculate_latest_metrics_with_both_data_types(self, query_handler):
        """Test calculate_latest_metrics with both metrics and results data."""
        connection = query_handler._connection
        metrics_table = query_handler.tables[mm_schemas.TimescaleDBTables.METRICS]
        results_table = query_handler.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]

        test_time = datetime(2024, 1, 15, 12, 0, 0)

        # Insert both metrics and results data for same application
        connection.run(
            statements=[
                f"""
                INSERT INTO {metrics_table.full_name()}
                (end_infer_time, start_infer_time, endpoint_id, application_name, metric_name, metric_value)
                VALUES ('{test_time}', '{test_time}', 'test_endpoint', 'test_app', 'accuracy', 0.95)
                """,
                f"""
                INSERT INTO {results_table.full_name()}
                (end_infer_time, start_infer_time, endpoint_id, application_name,
                 result_name, result_value, result_status, result_kind)
                VALUES ('{test_time}', '{test_time}', 'test_endpoint', 'test_app', 'drift_check', 0.1, 1, 1)
                """,
            ]
        )

        result = query_handler.calculate_latest_metrics(
            application_names=["test_app"],
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
        )

        # Should return records for both metrics and results
        assert isinstance(result, list)
        assert (
            len(result) >= 1
        )  # Should have at least one record (could be combined or separate)
