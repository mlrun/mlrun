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

from datetime import datetime

import pytest

import services.api.crud.alert_activation


@pytest.mark.parametrize(
    "partition_interval, partition_datetime, expected_name, expected_partition_value, expected_expression",
    [
        (
            "DAY",
            datetime(2024, 10, 30),
            "20241030",
            "20241031",
            "DAY(activation_time)",
        ),
        (
            "MONTH",
            datetime(2024, 10, 30),
            "202410",
            "202411",
            "MONTH(activation_time)",
        ),
        (
            "YEARWEEK",
            datetime(2024, 10, 30),
            "202444",
            "202445",
            "YEARWEEK(activation_time, 1)",
        ),
        (
            "YEARWEEK",
            datetime(2023, 1, 1),
            "202252",
            "202301",
            "YEARWEEK(activation_time, 1)",
        ),
        (
            "YEARWEEK",
            datetime(2024, 12, 31),
            "202501",
            "202502",
            "YEARWEEK(activation_time, 1)",
        ),
        (
            "YEARWEEK",
            datetime(2024, 1, 1),
            "202401",
            "202402",
            "YEARWEEK(activation_time, 1)",
        ),
        (
            "YEARWEEK",
            datetime(2024, 6, 15),
            "202424",
            "202425",
            "YEARWEEK(activation_time, 1)",
        ),
    ],
)
def test_get_partition_info_for_datetime(
    partition_interval,
    partition_datetime,
    expected_name,
    expected_partition_value,
    expected_expression,
):
    """
    To test from MySQL, use following code:
    `SELECT YEARWEEK('2024-12-31', 1) AS `yearweek_value`;`
    """
    # Get actual values from the function
    partition_name, partition_value, partition_expression = (
        services.api.crud.alert_activation.AlertActivation().get_partition_info(
            partition_interval,
            partition_datetime,
        )
    )

    # Assertions
    assert partition_name == expected_name
    assert partition_value == expected_partition_value
    assert partition_expression == expected_expression
