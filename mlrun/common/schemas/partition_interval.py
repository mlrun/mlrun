# Copyright 2024 Iguazio
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
import math
import os
from datetime import datetime, timedelta

import mlrun.common.types


class PartitionInterval(mlrun.common.types.StrEnum):
    DAY = "DAY"
    MONTH = "MONTH"
    YEARWEEK = "YEARWEEK"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls._value2member_map_

    @classmethod
    def valid_intervals(cls) -> list:
        return list(cls._value2member_map_.keys())

    def as_duration(self) -> timedelta:
        """
        Convert the partition interval to a duration-like timedelta.

        Returns:
            timedelta: A duration representing the partition interval.
        """
        if self == PartitionInterval.DAY:
            return timedelta(days=1)
        elif self == PartitionInterval.MONTH:
            # Approximate a month as 30 days
            return timedelta(days=30)
        elif self == PartitionInterval.YEARWEEK:
            return timedelta(weeks=1)
        else:
            raise ValueError(f"Unsupported PartitionInterval: {self}")

    def get_partition_names_and_boundaries(
        self,
        start_datetime: datetime,
        partitions_count: int = 1,
    ) -> list[tuple[str, int]]:
        """
        Generate partition names together with their corresponding RANGE partition
        boundaries, starting from a given datetime.

        This method is typically used when creating or extending RANGE-partitioned
        database tables (e.g. MySQL / PostgreSQL), where each partition is defined as:

            PARTITION <name> VALUES LESS THAN (<boundary_value>)

        For each partition:
        1. A partition name is derived from the *current* datetime
           (e.g. by day / month / yearweek, depending on the PartitionInterval).
        2. The next partition boundary datetime is calculated by advancing one
           partition interval forward.
        3. That boundary datetime is converted into an integer partition key value,
           which becomes the `VALUES LESS THAN` boundary for the current partition.
        4. The current datetime is advanced to the boundary, and the process repeats.

        This ensures that each partition fully covers its intended time range and
        that boundaries are strictly increasing.

        :param start_datetime:
            The datetime from which partition generation begins. This represents
            the *start* of the first partition's range.
        :param partitions_count:
            How many consecutive partitions to generate starting from
            `start_datetime`.

        :return:
            A list of `(partition_name, partition_boundary_value)` tuples, where:
            - `partition_name` is the generated partition name.
            - `partition_boundary_value` is the integer value used in
              `VALUES LESS THAN (...)` for that partition.
        """
        current_datetime = start_datetime
        partition_names_and_values = []
        for _ in range(partitions_count):
            partition_name = self.get_partition_name(
                current_datetime=current_datetime,
            )
            next_partition_boundary_date = self.get_next_partition_time(
                current_datetime=current_datetime,
            )
            next_partition_value = self.get_partition_key_value(
                current_datetime=next_partition_boundary_date,
            )
            partition_names_and_values.append((partition_name, next_partition_value))

            # Move to the next interval
            current_datetime = next_partition_boundary_date
        return partition_names_and_values

    def get_next_partition_time(
        self,
        current_datetime: datetime,
    ) -> datetime:
        """
        Calculates the next partition boundary time based on the specified partition interval.
        :param current_datetime: The current datetime from which the next interval is calculated.

        :return: A datetime object representing the start of the next partition interval.
            - If the interval is DAY, it advances by one day.
            - If the interval is MONTH, it advances to the first day of the next month.
            - If the interval is YEARWEEK, it advances by one week.
        """
        if self == PartitionInterval.DAY:
            return current_datetime + timedelta(days=1)
        elif self == PartitionInterval.MONTH:
            return (current_datetime.replace(day=1) + timedelta(days=32)).replace(day=1)
        elif self == PartitionInterval.YEARWEEK:
            return current_datetime + timedelta(weeks=1)
        else:
            raise ValueError(f"Unsupported PartitionInterval: {self}")

    def get_partition_key_value(
        self,
        current_datetime: datetime,
    ) -> int:
        format_string = PARTITION_INTERVAL_STRFTIME_FORMATS.get(self)
        if format_string is not None:
            return int(current_datetime.strftime(format_string))

        elif self == PartitionInterval.YEARWEEK:
            # Match MySQL YEARWEEK(date, 1):
            # ISO week-based year and week number.
            iso_year, iso_week, _ = current_datetime.isocalendar()
            return iso_year * 100 + iso_week

        else:
            raise ValueError(f"Unsupported PartitionInterval: {self}")

    def get_partition_name(
        self,
        current_datetime: datetime,
    ) -> str:
        return f"p{self.get_partition_key_value(current_datetime)}"

    def get_mysql_partition_key_sql(
        self,
        column_name: str,
    ) -> str:
        """
        Convert *column_name* into an integer partition key suitable for MySQL RANGE
        partitioning. Produces one of:
          - CAST(DATE_FORMAT(column_name, '%Y%m%d') AS UNSIGNED)
          - CAST(DATE_FORMAT(column_name, '%Y%m') AS UNSIGNED)
          - YEARWEEK(column_name, 1)
        """
        format_string = PARTITION_INTERVAL_STRFTIME_FORMATS.get(self)
        if format_string is not None:
            return f"CAST(DATE_FORMAT({column_name}, '{format_string}') AS UNSIGNED)"
        elif self == PartitionInterval.YEARWEEK:
            return f"YEARWEEK({column_name}, 1)"
        else:
            raise ValueError(f"Unsupported PartitionInterval: {self}")

    def get_number_of_partitions(
        self,
        days: int,
    ) -> int:
        # Calculate the number partitions based on given number of days
        if self == PartitionInterval.DAY:
            return days
        elif self == PartitionInterval.MONTH:
            # Average number days in a month is 30.44
            return math.ceil(days / 30.44)
        elif self == PartitionInterval.YEARWEEK:
            return math.ceil(days / 7)
        else:
            raise ValueError(f"Unsupported PartitionInterval: {self}")

    @classmethod
    def get_partition_interval_from_env(cls) -> "PartitionInterval":
        """
        Resolve the partition interval from an environment variable.

        This method intentionally reads the partition interval from the
        `PARTITION_INTERVAL` *environment variable* rather than from MLRun
        configuration for the following reasons:

        1. **Historic Alembic migrations**
           This logic is used by legacy / historic Alembic migration scripts that
           operate outside the normal MLRun runtime configuration flow. At migration
           time, MLRun configuration may not be fully initialized or available, while
           environment variables are guaranteed to be accessible.

        2. **QA-only override**
           `PARTITION_INTERVAL` is an *undocumented* environment variable intended
           solely for QA and testing scenarios. It allows tests to simulate different
           partitioning behaviors without changing production configuration or code.
           End users are not expected to set or rely on this value.

        3. **Explicit non-configurability**
           Partitioning strategy is a structural database concern and **not** a
           runtime-tunable configuration option. Reading this value from MLRun
           configuration would imply that users can change it dynamically and expect
           the partitioning scheme to change accordingly — which is not supported
           and would be unsafe for existing data.

        For these reasons, the value is:
        - Read once from the environment
        - Validated strictly against supported partition intervals
        - Intentionally disconnected from MLRun configuration mechanisms

        If the variable is not set, the default partition interval (`YEARWEEK`) is
        used.

        :raises ValueError:
            If `PARTITION_INTERVAL` is set to an unsupported value.
        """

        name = os.getenv("PARTITION_INTERVAL", "YEARWEEK").upper()
        if not PartitionInterval.is_valid(name):
            raise ValueError(
                f"PARTITION_INTERVAL must be one of {PartitionInterval.valid_intervals()}, got {name}"
            )
        return PartitionInterval(name)


PARTITION_INTERVAL_STRFTIME_FORMATS = {
    PartitionInterval.DAY: "%Y%m%d",
    PartitionInterval.MONTH: "%Y%m",
}
