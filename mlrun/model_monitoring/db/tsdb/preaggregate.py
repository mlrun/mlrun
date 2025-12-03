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
from typing import Any, Optional

import mlrun.errors
import mlrun.utils

# Compiled regex pattern for parsing time intervals (e.g., "1h", "10m", "1d", "1w", "1M")
_INTERVAL_PATTERN = re.compile(r"(\d+)([mhdwM])")


def _config_to_list(config_value: Any) -> list[str]:
    """Convert mlrun Config value to a list of strings.

    Handles mlrun.config.Config objects which store values in a _cfg dict.
    """
    if not config_value:
        return []
    if hasattr(config_value, "_cfg"):
        cfg = getattr(config_value, "_cfg", None)
        if cfg:
            return [str(v) for v in cfg.values()]
        return []
    try:
        return [str(v) for v in config_value]
    except TypeError:
        return []


def _config_to_dict(config_value: Any) -> dict[str, str]:
    """Convert mlrun Config value to a dict of strings.

    Handles mlrun.config.Config objects which have to_dict() or _cfg attribute.
    """
    if not config_value:
        return {}
    if hasattr(config_value, "to_dict"):
        return getattr(config_value, "to_dict")()
    if hasattr(config_value, "_cfg"):
        cfg = getattr(config_value, "_cfg", None)
        return {str(k): str(v) for k, v in cfg.items()} if cfg else {}
    try:
        return {str(k): str(v) for k, v in config_value.items()}
    except (TypeError, AttributeError):
        return {}


@dataclass
class PreAggregateConfig:
    """Configuration for pre-aggregated tables and retention policies."""

    aggregate_intervals: Optional[list[str]] = None
    agg_functions: Optional[list[str]] = None
    retention_policy: Optional[dict[str, str]] = None

    @classmethod
    def from_mlrun_config(cls) -> Optional["PreAggregateConfig"]:
        """
        Load pre-aggregate configuration from mlrun.mlconf.

        Reads the TSDB pre-aggregate configuration from the global MLRun config
        system, allowing configuration via environment variables or config files.

        :return: PreAggregateConfig if enabled in config, None if disabled
        :raises mlrun.errors.MLRunInvalidArgumentError: If config is malformed
        """
        import mlrun.config

        try:
            pre_agg_config = (
                mlrun.config.config.model_endpoint_monitoring.tsdb.pre_aggregate
            )
        except AttributeError as e:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Pre-aggregate config section not found in mlrun.mlconf: {e}"
            ) from e

        # Check if pre-aggregation is enabled
        enabled = pre_agg_config.enabled
        if isinstance(enabled, str):
            enabled = enabled.lower() in ("true", "1", "yes")

        if not enabled:
            return None

        # Convert Config objects to proper Python types
        aggregate_intervals = _config_to_list(pre_agg_config.aggregate_intervals)
        agg_functions = _config_to_list(pre_agg_config.agg_functions)
        retention_policy = _config_to_dict(pre_agg_config.retention_policy)

        # Validate required fields
        if not aggregate_intervals:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Pre-aggregate config missing 'aggregate_intervals'"
            )
        if not agg_functions:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Pre-aggregate config missing 'agg_functions'"
            )
        if not retention_policy:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Pre-aggregate config missing 'retention_policy'"
            )

        return cls(
            aggregate_intervals=aggregate_intervals,
            agg_functions=agg_functions,
            retention_policy=retention_policy,
        )


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

        intervals = self._pre_aggregate_config.aggregate_intervals or []
        if interval and interval not in intervals:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Interval '{interval}' not available in pre-aggregate configuration. "
                f"Available intervals: {intervals}"
            )

        functions = self._pre_aggregate_config.agg_functions or []
        if agg_function and agg_function not in functions:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Aggregation function '{agg_function}' not available in pre-aggregate configuration. "
                f"Available functions: {functions}"
            )

    def can_use_pre_aggregates(
        self, interval: Optional[str] = None, agg_funcs: Optional[list[str]] = None
    ) -> bool:
        """Check if pre-aggregates can be used for the given parameters."""
        if not self._pre_aggregate_config or not interval:
            return False

        intervals = self._pre_aggregate_config.aggregate_intervals or []
        if interval not in intervals:
            return False

        if agg_funcs:
            functions = self._pre_aggregate_config.agg_functions or []
            return all(func in functions for func in agg_funcs)

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
        if (
            not self._pre_aggregate_config
            or not self._pre_aggregate_config.aggregate_intervals
        ):
            return []
        return self._pre_aggregate_config.aggregate_intervals.copy()

    def get_available_functions(self) -> list[str]:
        """Get list of available aggregation functions."""
        if (
            not self._pre_aggregate_config
            or not self._pre_aggregate_config.agg_functions
        ):
            return []
        return self._pre_aggregate_config.agg_functions.copy()

    def get_retention_policy(self) -> dict[str, str]:
        """Get the retention policy configuration."""
        if (
            not self._pre_aggregate_config
            or not self._pre_aggregate_config.retention_policy
        ):
            return {}
        return self._pre_aggregate_config.retention_policy.copy()
