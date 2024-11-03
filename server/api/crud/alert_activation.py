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
    "YEARWEEK": "%Y%V",  # ISO week number (for weeks starting Monday)
    "MONTH": "%Y%m",  # Format as 'YYYYMM' (year and month)
    "DAY": "%Y%m%d",  # Format as 'YYYYMMDD' (year, month, day)
}


class AlertActivation(
    metaclass=mlrun.utils.singleton.Singleton,
):
    @staticmethod
    def get_partition_info(partition_interval, partition_datetime):
        """
        Generates partition details for a specified interval and datetime.

        :param partition_interval: The partitioning interval type, e.g., "DAY", "MONTH", or "YEARWEEK".
        :param partition_datetime: The datetime used for generating partition details.

        :return: A tuple containing:
            - partition_name: The name for the partition.
            - partition_value: The "LESS THAN" value for the next partition boundary.
            - partition_expression: The SQL partition expression.
        """
        partition_name = partition_datetime.strftime(
            partition_name_format_mapping[partition_interval]
        )

        if partition_interval == "DAY":
            partition_boundary_date = partition_datetime + timedelta(days=1)
        elif partition_interval == "MONTH":
            # Set to the first day of the next month
            partition_boundary_date = (
                partition_datetime.replace(day=1) + timedelta(days=32)
            ).replace(day=1)
        else:
            partition_boundary_date = partition_datetime + timedelta(weeks=1)

        if partition_interval == "YEARWEEK":
            partition_value = partition_boundary_date.strftime("%Y%V")
            partition_expression = "YEARWEEK(activation_time, 1)"
        else:
            partition_value = partition_boundary_date.strftime("%Y-%m-%d")
            partition_expression = f"{partition_interval}(activation_time)"
        return partition_name, partition_value, partition_expression
