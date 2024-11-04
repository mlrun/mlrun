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

import unittest.mock
from datetime import datetime

import pytest
import sqlalchemy.orm

import services.api.crud.alert_activation
import services.api.db.sqldb


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


@pytest.mark.parametrize(
    "partition_interval, retention_weeks, test_date, expected_cutoff_name",
    [
        ("DAY", 4, datetime(2024, 1, 1), "p20231204"),
        ("MONTH", 6, datetime(2024, 7, 15), "p202406"),
        ("YEARWEEK", 12, datetime(2024, 6, 1), "p202410"),
    ],
)
def test_drop_old_partitions(
    db: sqlalchemy.orm.Session,
    partition_interval,
    retention_weeks,
    test_date,
    expected_cutoff_name,
):
    with (
        unittest.mock.patch(
            "services.api.crud.alert_activation.datetime"
        ) as mock_datetime,
        unittest.mock.patch.object(
            services.api.crud.alert_activation.AlertActivation,
            "get_partition_expression_and_interval",
        ) as mocked_get_partition_expression_and_interval,
        unittest.mock.patch.object(
            services.api.utils.singletons.db.get_db(), "drop_partitions"
        ) as mocked_db_drop_partitions,
    ):
        mock_datetime.now.return_value = test_date

        mocked_get_partition_expression_and_interval.return_value = (
            "",
            partition_interval,
        )
        mocked_db_drop_partitions.return_value = None

        services.api.crud.alert_activation.AlertActivation().drop_old_partitions(
            db, retention_weeks
        )

        mocked_db_drop_partitions.assert_called_once_with(
            db, "alert_activation", expected_cutoff_name
        )


@pytest.mark.parametrize(
    "partition_interval, partition_number, test_date, expected_partition_info",
    [
        # Test cases with different partition intervals and partition numbers
        (
            "DAY",
            3,
            datetime(2024, 1, 1),
            [
                ("20240101", "20240102", "DAY(activation_time)"),
                ("20240102", "20240103", "DAY(activation_time)"),
                ("20240103", "20240104", "DAY(activation_time)"),
            ],
        ),
        (
            "MONTH",
            2,
            datetime(2024, 1, 1),
            [
                ("202401", "202402", "MONTH(activation_time)"),
                ("202402", "202403", "MONTH(activation_time)"),
            ],
        ),
        (
            "YEARWEEK",
            2,
            datetime(2024, 12, 31),
            [
                ("202501", "202502", "YEARWEEK(activation_time, 1)"),
                ("202502", "202503", "YEARWEEK(activation_time, 1)"),
            ],
        ),
    ],
)
def test_create_partitions(
    db: sqlalchemy.orm.Session,
    partition_interval,
    partition_number,
    test_date,
    expected_partition_info,
):
    with (
        unittest.mock.patch(
            "services.api.crud.alert_activation.datetime"
        ) as mock_datetime,
        unittest.mock.patch.object(
            services.api.crud.alert_activation.AlertActivation,
            "get_partition_expression_and_interval",
        ) as mocked_get_partition_expression_and_interval,
        unittest.mock.patch.object(
            services.api.utils.singletons.db.get_db(),
            "create_partitions",
        ) as mocked_db_create_partitions,
    ):
        mock_datetime.now.return_value = test_date

        # Mock return values for partition interval and info retrieval
        mocked_get_partition_expression_and_interval.return_value = (
            "",
            partition_interval,
        )

        services.api.crud.alert_activation.AlertActivation().create_partitions(
            db, partition_number
        )

        # Verify that create_partitions was called with the expected partition information
        mocked_db_create_partitions.assert_called_once_with(
            session=db,
            table_name="alert_activation",
            partitioning_information_list=expected_partition_info,
        )


@pytest.mark.parametrize(
    "mocked_partition_expression, expected_partition_interval",
    [
        ("month(`activation_time`)", "MONTH"),
        ("dayofmonth(`activation_time`)", "DAY"),
        ("yearweek(`activation_time`, 1)", "YEARWEEK"),
    ],
)
def test_get_partition_expression_and_interval(
    db: sqlalchemy.orm.Session,
    mocked_partition_expression,
    expected_partition_interval,
):
    with (
        unittest.mock.patch.object(
            services.api.utils.singletons.db.get_db(),
            "get_partition_expression_for_table",
        ) as mocked_get_partition_expression_for_table,
    ):
        mocked_get_partition_expression_for_table.return_value = mocked_partition_expression
        partition_expression, partition_interval = (
            services.api.crud.alert_activation.AlertActivation().get_partition_expression_and_interval(db)
        )

        assert partition_expression == mocked_partition_expression
        assert partition_interval == expected_partition_interval
