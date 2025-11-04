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

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import mlrun.errors
import mlrun.utils
from mlrun.model_monitoring.db.tsdb.preaggregate import (
    PreAggregateManager,
)


class MockPreAggregateConfig:
    """Mock configuration for testing."""

    def __init__(self, intervals=None, functions=None):
        self.aggregate_intervals = intervals or ["10m", "1h", "6h", "1d", "1w", "1M"]
        self.agg_functions = functions or ["sum", "avg", "min", "max", "count", "last"]


class TestPreAggregateManager:
    """Test suite for PreAggregateManager class."""

    def test_init_with_config(self):
        """Test initialization with pre-aggregate config."""
        config = MockPreAggregateConfig()
        handler = PreAggregateManager(config)
        assert handler._pre_aggregate_config == config

    def test_init_without_config(self):
        """Test initialization without pre-aggregate config."""
        handler = PreAggregateManager()
        assert handler._pre_aggregate_config is None

    def test_validate_interval_and_function_no_params(self):
        """Test validation when no interval or function provided."""
        handler = PreAggregateManager()
        # Should not raise any exception
        handler.validate_interval_and_function(None, None)

    def test_validate_interval_and_function_no_config(self):
        """Test validation fails when no config but params provided."""
        handler = PreAggregateManager()

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=r"Pre-aggregate configuration not available",
        ):
            handler.validate_interval_and_function("1h", None)

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=r"Pre-aggregate configuration not available",
        ):
            handler.validate_interval_and_function(None, "avg")

    def test_validate_interval_and_function_invalid_interval(self):
        """Test validation fails for invalid interval."""
        config = MockPreAggregateConfig(intervals=["1h", "1d"])
        handler = PreAggregateManager(config)

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=r"Interval '5m' not available.*Available intervals: \['1h', '1d'\]",
        ):
            handler.validate_interval_and_function("5m", "avg")

    def test_validate_interval_and_function_invalid_function(self):
        """Test validation fails for invalid aggregation function."""
        config = MockPreAggregateConfig(functions=["sum", "avg"])
        handler = PreAggregateManager(config)

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=r"Aggregation function 'median' not available.*Available functions: \['sum', 'avg'\]",
        ):
            handler.validate_interval_and_function("1h", "median")

    def test_validate_interval_and_function_valid_params(self):
        """Test validation passes for valid parameters."""
        config = MockPreAggregateConfig()
        handler = PreAggregateManager(config)

        # Should not raise any exception
        handler.validate_interval_and_function("1h", "avg")
        handler.validate_interval_and_function("1d", None)
        handler.validate_interval_and_function(None, "sum")

    def test_can_use_pre_aggregates_no_config(self):
        """Test can_use_pre_aggregates returns False when no config."""
        handler = PreAggregateManager()
        assert not handler.can_use_pre_aggregates("1h", ["avg"])

    def test_can_use_pre_aggregates_no_interval(self):
        """Test can_use_pre_aggregates returns False when no interval."""
        config = MockPreAggregateConfig()
        handler = PreAggregateManager(config)
        assert not handler.can_use_pre_aggregates(None, ["avg"])

    def test_can_use_pre_aggregates_invalid_interval(self):
        """Test can_use_pre_aggregates returns False for invalid interval."""
        config = MockPreAggregateConfig(intervals=["1h"])
        handler = PreAggregateManager(config)
        assert not handler.can_use_pre_aggregates("5m", ["avg"])

    def test_can_use_pre_aggregates_invalid_functions(self):
        """Test can_use_pre_aggregates returns False for invalid functions."""
        config = MockPreAggregateConfig(functions=["sum", "avg"])
        handler = PreAggregateManager(config)
        assert not handler.can_use_pre_aggregates("1h", ["median"])
        assert not handler.can_use_pre_aggregates("1h", ["avg", "median"])

    def test_can_use_pre_aggregates_valid_params(self):
        """Test can_use_pre_aggregates returns True for valid parameters."""
        config = MockPreAggregateConfig()
        handler = PreAggregateManager(config)

        assert handler.can_use_pre_aggregates("1h")
        assert handler.can_use_pre_aggregates("1h", ["avg"])
        assert handler.can_use_pre_aggregates("1h", ["sum", "avg"])

    def test_align_time_to_interval_no_interval(self):
        """Test time alignment when no interval provided."""
        handler = PreAggregateManager()
        dt = datetime(2025, 1, 15, 14, 35, 42)

        result = handler.align_time_to_interval(dt, None)
        assert result == dt

    def test_align_time_to_interval_invalid_format(self):
        """Test time alignment with invalid interval format."""
        handler = PreAggregateManager()
        dt = datetime(2025, 1, 15, 14, 35, 42)

        result = handler.align_time_to_interval(dt, "invalid")
        assert result == dt

    def test_align_time_to_interval_minutes(self):
        """Test time alignment for minute intervals."""
        handler = PreAggregateManager()
        dt = datetime(2025, 1, 15, 14, 37, 42)  # 37 minutes, 42 seconds

        # Align start (round down)
        result = handler.align_time_to_interval(dt, "10m", align_start=True)
        expected = datetime(2025, 1, 15, 14, 30, 0)  # Round down to 30 minutes
        assert result == expected

        # Align end (round up)
        result = handler.align_time_to_interval(dt, "10m", align_start=False)
        expected = datetime(2025, 1, 15, 14, 40, 0)  # Round up to 40 minutes
        assert result == expected

    def test_align_time_to_interval_hours(self):
        """Test time alignment for hour intervals."""
        handler = PreAggregateManager()
        dt = datetime(2025, 1, 15, 14, 35, 42)  # 2 PM, 35 minutes

        # Align start (round down)
        result = handler.align_time_to_interval(dt, "6h", align_start=True)
        expected = datetime(2025, 1, 15, 12, 0, 0)  # Round down to 12 PM (14 % 6 = 2)
        assert result == expected

        # Align end (round up)
        result = handler.align_time_to_interval(dt, "6h", align_start=False)
        expected = datetime(2025, 1, 15, 18, 0, 0)  # Round up to 6 PM
        assert result == expected

    def test_align_time_to_interval_days(self):
        """Test time alignment for day intervals."""
        handler = PreAggregateManager()
        dt = datetime(2025, 1, 15, 14, 35, 42)

        # Align start (round down)
        result = handler.align_time_to_interval(dt, "1d", align_start=True)
        expected = datetime(2025, 1, 15, 0, 0, 0)  # Start of day
        assert result == expected

        # Align end (round up)
        result = handler.align_time_to_interval(dt, "1d", align_start=False)
        expected = datetime(2025, 1, 16, 0, 0, 0)  # Start of next day
        assert result == expected

    def test_align_time_to_interval_months(self):
        """Test time alignment for month intervals."""
        handler = PreAggregateManager()
        dt = datetime(2025, 6, 15, 14, 35, 42)  # June 15th

        # Align start (round down)
        result = handler.align_time_to_interval(dt, "1M", align_start=True)
        expected = datetime(2025, 6, 1, 0, 0, 0)  # Start of June
        assert result == expected

        # Align end (round up)
        result = handler.align_time_to_interval(dt, "1M", align_start=False)
        expected = datetime(2025, 7, 1, 0, 0, 0)  # Start of July
        assert result == expected

    def test_align_time_to_interval_months_december(self):
        """Test time alignment for months when crossing year boundary."""
        handler = PreAggregateManager()
        dt = datetime(2025, 12, 15, 14, 35, 42)  # December 15th

        # Align end (round up) - should go to next year
        result = handler.align_time_to_interval(dt, "1M", align_start=False)
        expected = datetime(2026, 1, 1, 0, 0, 0)  # Start of January next year
        assert result == expected

    def test_align_time_range(self):
        """Test aligning both start and end times."""
        handler = PreAggregateManager()
        start = datetime(2025, 1, 15, 14, 37, 42)
        end = datetime(2025, 1, 15, 16, 23, 18)

        # Test with interval
        aligned_start, aligned_end = handler.align_time_range(start, end, "1h")
        assert aligned_start == datetime(2025, 1, 15, 14, 0, 0)  # Start of 2 PM hour
        assert aligned_end == datetime(2025, 1, 15, 17, 0, 0)  # Start of 5 PM hour

        # Test without interval
        result_start, result_end = handler.align_time_range(start, end, None)
        assert result_start == start
        assert result_end == end

    @patch("mlrun.utils.datetime_min")
    @patch("mlrun.utils.datetime_now")
    def test_get_start_end_with_none_values(self, mock_now, mock_min):
        """Test get_start_end with None values."""
        mock_min_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_now_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        mock_min.return_value = mock_min_time
        mock_now.return_value = mock_now_time

        # Test with both None
        start, end = PreAggregateManager.get_start_end(None, None)
        assert start == mock_min_time
        assert end == mock_now_time

        # Test with start provided
        provided_start = datetime(2025, 1, 10, 10, 0, 0)
        start, end = PreAggregateManager.get_start_end(provided_start, None)
        assert start == provided_start
        assert end == mock_now_time

        # Test with end provided
        provided_end = datetime(2025, 1, 20, 15, 0, 0)
        start, end = PreAggregateManager.get_start_end(None, provided_end)
        assert start == mock_min_time
        assert end == provided_end

    def test_get_start_end_with_datetime_values(self):
        """Test get_start_end with provided datetime values."""
        start_time = datetime(2025, 1, 10, 10, 0, 0)
        end_time = datetime(2025, 1, 15, 15, 0, 0)

        start, end = PreAggregateManager.get_start_end(start_time, end_time)
        assert start == start_time
        assert end == end_time

    def test_get_start_end_invalid_types(self):
        """Test get_start_end raises error for invalid types."""
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=r"Both start and end must be datetime objects",
        ):
            PreAggregateManager.get_start_end("2025-01-01", None)

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=r"Both start and end must be datetime objects",
        ):
            PreAggregateManager.get_start_end(None, "2025-01-15")

    def test_multiple_interval_formats(self):
        """Test various interval format edge cases."""
        handler = PreAggregateManager()
        dt = datetime(2025, 1, 15, 14, 35, 42)

        # Test different minute intervals
        result = handler.align_time_to_interval(dt, "5m", align_start=True)
        expected = datetime(2025, 1, 15, 14, 35, 0)  # 35 % 5 = 0, so 35 minutes
        assert result == expected

        result = handler.align_time_to_interval(dt, "15m", align_start=True)
        expected = datetime(2025, 1, 15, 14, 30, 0)  # 35 % 15 = 5, so round down to 30
        assert result == expected

        # Test different hour intervals
        dt_3pm = datetime(2025, 1, 15, 15, 35, 42)  # 3 PM
        result = handler.align_time_to_interval(dt_3pm, "4h", align_start=True)
        expected = datetime(2025, 1, 15, 12, 0, 0)  # 15 % 4 = 3, so round down to 12
        assert result == expected

    def test_integration_with_config_validation(self):
        """Test integration between config validation and time alignment."""
        config = MockPreAggregateConfig(
            intervals=["10m", "1h"], functions=["avg", "sum"]
        )
        handler = PreAggregateManager(config)

        # Valid scenario
        handler.validate_interval_and_function("10m", "avg")
        assert handler.can_use_pre_aggregates("10m", ["avg"])

        # Test time alignment for valid interval
        dt = datetime(2025, 1, 15, 14, 37, 42)
        aligned = handler.align_time_to_interval(dt, "10m", align_start=True)
        expected = datetime(2025, 1, 15, 14, 30, 0)
        assert aligned == expected

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        handler = PreAggregateManager()

        # Test exact alignment (no change needed)
        dt = datetime(2025, 1, 15, 14, 0, 0)  # Exactly on hour boundary
        result = handler.align_time_to_interval(dt, "1h", align_start=True)
        assert result == dt

        # Test month boundary edge case
        dt = datetime(2025, 2, 28, 23, 59, 59)  # End of February
        result = handler.align_time_to_interval(dt, "1M", align_start=False)
        expected = datetime(2025, 3, 1, 0, 0, 0)
        assert result == expected
