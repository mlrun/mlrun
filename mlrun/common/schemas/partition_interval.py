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
        Returns a list of partition details for a specified number of partitions starting from a given datetime.

        :param start_datetime: The starting datetime used for generating partition details.
        :param partitions_count: The number of partitions to generate details for.

        :return: A list of tuples:
            - partition_name: The name for the partition.
            - partition_value: The "LESS THAN" value for the next partition boundary.
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

        if self == PartitionInterval.YEARWEEK:
            year, week, _ = current_datetime.isocalendar()
            return int(f"{year}{week:02d}")

        raise ValueError(f"Unsupported PartitionInterval: {self}")

    def get_partition_name(
        self,
        current_datetime: datetime,
    ) -> str:
        return f"p{self.get_partition_key_value(current_datetime)}"

    def get_mysql_partition_key_sql(self, column_name: str) -> str:
        """
        Convert *column_name* into an integer partition key suitable for MySQL RANGE
        partitioning. Produces one of:
          - CAST(DATE_FORMAT(column_name, '%Y%m%d') AS UNSIGNED)
          - CAST(DATE_FORMAT(column_name, '%Y%m') AS UNSIGNED)
          - YEARWEEK(column_name, 3)
        """
        format_string = PARTITION_INTERVAL_STRFTIME_FORMATS.get(self)
        if format_string is not None:
            return f"CAST(DATE_FORMAT({column_name}, '{format_string}') AS UNSIGNED)"

        if self == PartitionInterval.YEARWEEK:
            # mode=3 → ISO-like weeks, matches datetime.isocalendar()
            return f"YEARWEEK({column_name}, 3)"

        raise ValueError(f"Unsupported PartitionInterval: {self}")

    def get_number_of_partitions(self, days: int) -> int:
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
        Parse PARTITION_INTERVAL once, validate, then cache.
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
