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

from datetime import timedelta

import mlrun.utils.singleton

valid_partition_intervals = ["DAY", "YEARWEEK", "MONTH"]
partition_name_format_mapping = {
    "MONTH": "%Y%m",  # Format as 'YYYYMM' (year and month)
    "DAY": "%Y%m%d",  # Format as 'YYYYMMDD' (year, month, day)
}


class AlertActivation(
    metaclass=mlrun.utils.singleton.Singleton,
):
    @staticmethod
    def get_partition_info(
        partition_interval, partition_datetime
    ) -> Tuple[str, str, str]:
        """
        Generates partition details for a specified interval and datetime.

        :param partition_interval: The partitioning interval type, e.g., "DAY", "MONTH", or "YEARWEEK".
        :param partition_datetime: The datetime used for generating partition details.

        :return: A tuple containing:
            - partition_name: The name for the partition.
            - partition_value: The "LESS THAN" value for the next partition boundary.
            - partition_expression: The SQL partition expression.
        """
        if partition_interval == "YEARWEEK":
            year, week, _ = partition_datetime.isocalendar()
            partition_name = f"{year}{week:02d}"

            next_week = partition_datetime + timedelta(weeks=1)
            next_year, next_week_num, _ = next_week.isocalendar()
            partition_value = f"{next_year}{next_week_num:02d}"

            partition_expression = "YEARWEEK(activation_time, 1)"
            return partition_name, partition_value, partition_expression

        partition_name = partition_datetime.strftime(
            partition_name_format_mapping[partition_interval]
        )

        if partition_interval == "DAY":
            partition_boundary_date = partition_datetime + timedelta(days=1)
            partition_value = partition_boundary_date.strftime(
                partition_name_format_mapping["DAY"]
            )
            partition_expression = "DAY(activation_time)"
        elif partition_interval == "MONTH":
            partition_boundary_date = (
                partition_datetime.replace(day=1) + timedelta(days=32)
            ).replace(day=1)
            partition_value = partition_boundary_date.strftime(
                partition_name_format_mapping["MONTH"]
            )
            partition_expression = "MONTH(activation_time)"
        else:
            raise ValueError(f"Unsupported partition interval: {partition_interval}")

        return partition_name, partition_value, partition_expression
