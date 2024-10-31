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


import mlrun.utils.singleton

valid_partition_intervals = ["DAY", "YEARWEEK", "MONTH"]
partition_name_format_mapping = {
    "YEARWEEK": "%Y%V",  # ISO week number (for weeks starting Monday)
    "MONTH": "%Y%m",  # Format as 'YYYYMM' (year and month)
    "DAY": "%Y%m%d",  # Format as 'YYYYMMDD' (year, month, day)
}


class AlertHistory(
    metaclass=mlrun.utils.singleton.Singleton,
):
    @staticmethod
    def get_partition_info_for_datetime(partition_interval, partition_datetime):
        partition_name = partition_datetime.strftime(
            partition_name_format_mapping[partition_interval]
        )

        if partition_interval == "YEARWEEK":
            partition_value = int(
                partition_datetime.strftime("%Y%V")
            )  # Example: 202444 for 44th week of 2024
            partition_expression = "YEARWEEK(activation_time, 1)"
        else:
            partition_value = partition_datetime.strftime("%Y-%m-%d")
            partition_expression = f"{partition_interval}(activation_time)"
        return partition_name, partition_value, partition_expression
