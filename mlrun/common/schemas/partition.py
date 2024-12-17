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
#
from datetime import datetime, timedelta

from mlrun.common.types import StrEnum


class PartitionInterval(StrEnum):
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

    @classmethod
    def from_expression(cls, partition_expression: str):
        """
        Returns the corresponding PartitionInterval for a given partition expression,
        or None if the function is not mapped.

        :param partition_expression: The partition expression to map to an interval.
        :return: PartitionInterval corresponding to the expression, or `month` if no match is found.
        """

        # Match the provided function string to the correct interval
        partition_expression = partition_expression.upper()
        if "YEARWEEK" in partition_expression:
            return cls.YEARWEEK
        elif "DAYOFMONTH" in partition_expression:
            return cls.DAY
        else:
            return cls.MONTH

    def get_partition_info(
        self,
        start_datetime: datetime,
        partition_number: int = 1,
    ) -> list[tuple[str, str]]:
        """
        Generates partition details for a specified number of partitions starting from a given datetime.

        :param start_datetime: The starting datetime used for generating partition details.
        :param partition_number: The number of partitions to generate details for.

        :return: A list of tuples:
            - partition_name: The name for the partition.
            - partition_value: The "LESS THAN" value for the next partition boundary.
        """
        partitioning_information_list = []
        current_datetime = start_datetime

        for _ in range(partition_number):
            partition_name = self.get_partition_name(current_datetime)
            partition_boundary_date = self.get_next_partition_time(current_datetime)
            partition_value = self.get_partition_name(partition_boundary_date)
            partitioning_information_list.append((partition_name, partition_value))

            # Move to the next interval
            current_datetime = partition_boundary_date

        return partitioning_information_list

    def get_next_partition_time(self, current_datetime: datetime) -> datetime:
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

    def get_partition_name(self, current_datetime: datetime) -> str:
        if self == PartitionInterval.DAY:
            return current_datetime.strftime("%Y%m%d")
        elif self == PartitionInterval.MONTH:
            return current_datetime.strftime("%Y%m")
        elif self == PartitionInterval.YEARWEEK:
            year, week, _ = current_datetime.isocalendar()
            return f"{year}{week:02d}"

    def get_partition_expression(self, column_name: str):
        if self == PartitionInterval.YEARWEEK:
            return f"YEARWEEK({column_name}, 1)"
        elif self == PartitionInterval.DAY:
            # generates value in format %Y%m%d in mysql
            # mysql query example: `select YEAR(NOW())*10000 + MONTH(NOW())*100 + DAY(NOW());`
            return f"YEAR({column_name}) * 10000 + MONTH({column_name}) * 100 + DAY({column_name})"
        elif self == PartitionInterval.MONTH:
            # generates value in format %Y%m in mysql
            # mysql query example: `select YEAR(NOW())*100 + MONTH(NOW());`
            return f"YEAR({column_name}) * 100 + MONTH({column_name})"

    def get_number_of_partitions(self, days: int) -> int:
        # Calculate the number partitions based on given number of days
        if self == PartitionInterval.DAY:
            return days
        elif self == PartitionInterval.MONTH:
            # Average number days in a month is 30.44
            return int(days / 30.44)
        elif self == PartitionInterval.YEARWEEK:
            return int(days / 7)
