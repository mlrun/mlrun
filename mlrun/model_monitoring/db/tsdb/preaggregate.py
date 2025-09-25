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
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import mlrun.errors
import mlrun.utils

# Compiled regex pattern for parsing time intervals (e.g., "1h", "10m", "1d", "1w", "1M")
_INTERVAL_PATTERN = re.compile(r"(\d+)([mhdwM])")


@dataclass
class PreAggregateConfig:
    """Configuration for pre-aggregated tables and retention policies."""

    aggregate_intervals: list[str] = None
    agg_functions: list[str] = None
    retention_policy: dict[str, str] = None

    def __post_init__(self):
        if self.aggregate_intervals is None:
            self.aggregate_intervals = ["10m", "1h", "6h", "1d", "1w", "1M"]

        if self.agg_functions is None:
            self.agg_functions = ["sum", "avg", "min", "max", "count", "last"]

        if self.retention_policy is None:
            self.retention_policy = {
                "raw": "7d",
                "10m": "30d",
                "1h": "1y",
                "6h": "1y",
                "1d": "5y",
                "1w": "5y",
                "1M": "5y",
            }


class PreAggregateManager:
    """Handles pre-aggregate validation, time alignment, and optimization decisions."""

    def __init__(self, pre_aggregate_config: Optional[PreAggregateConfig] = None):
        """
        Initialize the pre-aggregate handler.

        :param pre_aggregate_config: Configuration for pre-aggregated tables and operations.
                                   If None, all pre-aggregate operations will be disabled.
        """
        self._pre_aggregate_config = pre_aggregate_config

    def validate_interval_and_function(
        self, interval: Optional[str], agg_function: Optional[str]
    ) -> None:
        """Validate that interval and aggregation function are available in pre-aggregate config."""
        if not interval and not agg_function:
            return

        if not self._pre_aggregate_config:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Pre-aggregate configuration not available. Cannot use interval or agg_function parameters."
            )

        if interval and interval not in self._pre_aggregate_config.aggregate_intervals:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Interval '{interval}' not available in pre-aggregate configuration. "
                f"Available intervals: {self._pre_aggregate_config.aggregate_intervals}"
            )

        if (
            agg_function
            and agg_function not in self._pre_aggregate_config.agg_functions
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Aggregation function '{agg_function}' not available in pre-aggregate configuration. "
                f"Available functions: {self._pre_aggregate_config.agg_functions}"
            )

    def can_use_pre_aggregates(
        self, interval: Optional[str] = None, agg_funcs: Optional[list[str]] = None
    ) -> bool:
        """Check if pre-aggregates can be used for the given parameters."""
        if not self._pre_aggregate_config or not interval:
            return False

        if interval not in self._pre_aggregate_config.aggregate_intervals:
            return False

        if agg_funcs:
            return all(
                func in self._pre_aggregate_config.agg_functions for func in agg_funcs
            )

        return True

    def align_time_to_interval(
        self, dt: datetime, interval: str, align_start: bool = True
    ) -> datetime:
        """Align datetime to interval boundaries."""
        if not interval:
            return dt

        # Parse interval (e.g., "1h", "10m", "1d")
        match = _INTERVAL_PATTERN.match(interval)
        if not match:
            return dt

        amount, unit = int(match.group(1)), match.group(2)

        # Get the start boundary for this interval
        aligned_start = self._get_interval_start_boundary(dt, amount, unit)

        if align_start:
            return aligned_start

        # For end alignment, add the interval duration to the start
        return self._add_interval_to_datetime(aligned_start, amount, unit)

    def _get_interval_start_boundary(
        self, dt: datetime, amount: int, unit: str
    ) -> datetime:
        """Get the start boundary for the given interval."""
        if unit == "m":  # minutes
            return dt.replace(second=0, microsecond=0) - timedelta(
                minutes=dt.minute % amount
            )
        elif unit == "h":  # hours
            return dt.replace(minute=0, second=0, microsecond=0) - timedelta(
                hours=dt.hour % amount
            )
        elif unit == "d":  # days
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif unit == "w":  # weeks
            # Align to Monday (start of week)
            days_since_monday = dt.weekday()
            return (dt - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif unit == "M":  # months (approximate)
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        return dt

    def _add_interval_to_datetime(
        self, dt: datetime, amount: int, unit: str
    ) -> datetime:
        """Add the specified interval amount to a datetime."""
        if unit == "m":  # minutes
            return dt + timedelta(minutes=amount)
        elif unit == "h":  # hours
            return dt + timedelta(hours=amount)
        elif unit == "d":  # days
            return dt + timedelta(days=amount)
        elif unit == "w":  # weeks
            return dt + timedelta(weeks=amount)
        elif unit == "M":  # months (approximate)
            if dt.month == 12:
                return dt.replace(year=dt.year + 1, month=1)
            return dt.replace(month=dt.month + 1)

        return dt

    def align_time_range(
        self, start: datetime, end: datetime, interval: Optional[str]
    ) -> tuple[datetime, datetime]:
        """Align both start and end times to interval boundaries."""
        if not interval:
            return start, end

        aligned_start = self.align_time_to_interval(start, interval, align_start=True)
        aligned_end = self.align_time_to_interval(end, interval, align_start=False)

        return aligned_start, aligned_end

    @staticmethod
    def get_start_end(
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> tuple[datetime, datetime]:
        """
        Utility function for TSDB start/end format validation.

        :param start: Either None or datetime, None is handled as datetime.min(tz=timezone.utc)
        :param end: Either None or datetime, None is handled as datetime.now(tz=timezone.utc)
        :return: start datetime, end datetime
        """
        start = start or mlrun.utils.datetime_min()
        end = end or mlrun.utils.datetime_now()
        if not (isinstance(start, datetime) and isinstance(end, datetime)):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Both start and end must be datetime objects"
            )
        return start, end

    @property
    def config(self) -> Optional[PreAggregateConfig]:
        """Get the current pre-aggregate configuration."""
        return self._pre_aggregate_config

    def is_pre_aggregates_enabled(self) -> bool:
        """Check if pre-aggregates are enabled (config is provided)."""
        return self._pre_aggregate_config is not None

    def get_available_intervals(self) -> list[str]:
        """Get list of available intervals for pre-aggregation."""
        if not self._pre_aggregate_config:
            return []
        return self._pre_aggregate_config.aggregate_intervals.copy()

    def get_available_functions(self) -> list[str]:
        """Get list of available aggregation functions."""
        if not self._pre_aggregate_config:
            return []
        return self._pre_aggregate_config.agg_functions.copy()

    def get_retention_policy(self) -> dict[str, str]:
        """Get the retention policy configuration."""
        if not self._pre_aggregate_config:
            return {}
        return self._pre_aggregate_config.retention_policy.copy()
