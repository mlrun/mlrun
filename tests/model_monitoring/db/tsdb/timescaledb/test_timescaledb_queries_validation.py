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

import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors

# Constants for expected error messages
AGGREGATION_PARAMS_ERROR_MSG = (
    "both or neither of `aggregation_window` and `agg_funcs` must be provided"
)


class TestParameterValidation:
    """Tests for parameter validation and error handling."""

    def test_read_predictions_invalid_aggregation_params(self, query_test_helper):
        """Test MLRunInvalidArgumentError when agg_funcs without aggregation_window."""
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()

        # Create predictions handler using test helper
        predictions_handler = query_test_helper.create_predictions_handler()

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
            predictions_handler.read_predictions(
                endpoint_id="test_endpoint",
                start=start_time,
                end=end_time,
                agg_funcs=["avg"],  # agg_funcs provided but no aggregation_window
            )

        assert str(exc_info.value) == AGGREGATION_PARAMS_ERROR_MSG, (
            f"Expected exact error message '{AGGREGATION_PARAMS_ERROR_MSG}', "
            f"but got: {exc_info.value}"
        )

    def test_read_predictions_invalid_aggregation_params_reverse(
        self, query_test_helper
    ):
        """Test MLRunInvalidArgumentError when aggregation_window without agg_funcs."""
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()

        # Create predictions handler using test helper
        predictions_handler = query_test_helper.create_predictions_handler()

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
            predictions_handler.read_predictions(
                endpoint_id="test_endpoint",
                start=start_time,
                end=end_time,
                aggregation_window="1h",  # aggregation_window provided but no agg_funcs
            )

        assert str(exc_info.value) == AGGREGATION_PARAMS_ERROR_MSG, (
            f"Expected exact error message '{AGGREGATION_PARAMS_ERROR_MSG}', "
            f"but got: {exc_info.value}"
        )

    def test_get_endpoint_filter_invalid_type(self, query_builder):
        """Test MLRunInvalidArgumentError for invalid endpoint_id type."""
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
            query_builder.build_endpoint_filter(
                123
            )  # Invalid type - should be string or list

        assert (
            str(exc_info.value)
            == "Invalid 'endpoint_ids' filter: must be a string or a list of strings"
        )

    def test_get_endpoint_filter_dict_type(self, query_builder):
        """Test MLRunInvalidArgumentError for dict endpoint_id type."""
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
            query_builder.build_endpoint_filter(
                {"endpoint": "test"}
            )  # Invalid type - should be string or list

        assert (
            str(exc_info.value)
            == "Invalid 'endpoint_ids' filter: must be a string or a list of strings"
        )


class TestPreAggregateExceptionHandling:
    """Tests for pre-aggregate exception handling and fallback logic."""

    def test_calculate_latest_metrics_with_error_data_only(self, query_test_helper):
        """Test calculate_latest_metrics with only error data (no metrics or results)."""

        metrics_handler = query_test_helper.create_metrics_handler()

        # Insert error data only - this tests edge case handling
        connection = query_test_helper.connection
        errors_table = query_test_helper.table_schemas[
            mm_schemas.TimescaleDBTables.ERRORS
        ]

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

        result = metrics_handler.calculate_latest_metrics(
            application_names=["test_app"],
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
        )

        assert len(result) == 0

    def test_calculate_latest_metrics_with_both_data_types(self, query_test_helper):
        """Test calculate_latest_metrics with both metrics and results data."""
        metrics_handler = query_test_helper.create_metrics_handler()

        connection = query_test_helper.connection
        metrics_table = query_test_helper.table_schemas[
            mm_schemas.TimescaleDBTables.METRICS
        ]
        results_table = query_test_helper.table_schemas[
            mm_schemas.TimescaleDBTables.APP_RESULTS
        ]

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

        result = metrics_handler.calculate_latest_metrics(
            application_names=["test_app"],
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
        )

        # Should return records for both metrics and results
        assert isinstance(result, list)
        assert (
            len(result) >= 1
        )  # Should have at least one record (could be combined or separate)
