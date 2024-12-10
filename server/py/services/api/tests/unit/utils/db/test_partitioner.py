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

import mlrun.common.schemas.alert
import mlrun.common.schemas.partition

import framework.utils.singletons.db
import services.api.utils.db.partitioner


@pytest.mark.parametrize(
    "partition_interval, partition_datetime, expected_name, expected_partition_value",
    [
        (
            "DAY",
            datetime(2024, 10, 30),
            "20241030",
            "20241031",
        ),
        (
            "MONTH",
            datetime(2024, 10, 30),
            "202410",
            "202411",
        ),
        (
            "YEARWEEK",
            datetime(2024, 10, 30),
            "202444",
            "202445",
        ),
        (
            "YEARWEEK",
            datetime(2023, 1, 1),
            "202252",
            "202301",
        ),
        (
            "YEARWEEK",
            datetime(2024, 12, 31),
            "202501",
            "202502",
        ),
        (
            "YEARWEEK",
            datetime(2024, 1, 1),
            "202401",
            "202402",
        ),
        (
            "YEARWEEK",
            datetime(2024, 6, 15),
            "202424",
            "202425",
        ),
    ],
)
def test_get_partition_info_for_datetime(
    partition_interval,
    partition_datetime,
    expected_name,
    expected_partition_value,
):
    """
    To test from MySQL, use following code:
    `SELECT YEARWEEK('2024-12-31', 1) AS `yearweek_value`;`
    """
    # Get actual values from the function
    partition_info = (
        mlrun.common.schemas.partition.PartitionInterval(
            partition_interval
        ).get_partition_info(
            partition_datetime,
        )
    )[0]

    # Assertions
    assert partition_info[0] == expected_name
    assert partition_info[1] == expected_partition_value


@pytest.mark.parametrize(
    "partition_interval, retention_days, test_date, expected_cutoff_name",
    [
        ("DAY", 4 * 7, datetime(2024, 1, 1), "p20231204"),
        ("DAY", 1, datetime(2024, 1, 1), "p20231231"),
        ("MONTH", 6 * 7, datetime(2024, 7, 15), "p202406"),
        ("YEARWEEK", 12 * 7, datetime(2024, 6, 1), "p202410"),
        ("YEARWEEK", 14 * 7, datetime(2024, 6, 1), "p202408"),
        ("YEARWEEK", 14 * 7, datetime(2024, 6, 1), "p202408"),
    ],
)
def test_drop_old_partitions(
    db: sqlalchemy.orm.Session,
    partition_interval,
    retention_days,
    test_date,
    expected_cutoff_name,
):
    with (
        unittest.mock.patch(
            "services.api.utils.db.partitioner.datetime"
        ) as mock_datetime,
        unittest.mock.patch.object(
            framework.utils.singletons.db.get_db(), "drop_partitions"
        ) as mocked_db_drop_partitions,
    ):
        mock_datetime.now.return_value = test_date
        mocked_db_drop_partitions.return_value = None

        services.api.utils.db.partitioner.MySQLPartitioner().drop_partitions(
            db,
            "alert_activations",
            retention_days,
            mlrun.common.schemas.partition.PartitionInterval(partition_interval),
        )

        mocked_db_drop_partitions.assert_called_once_with(
            db, "alert_activations", expected_cutoff_name
        )


@pytest.mark.parametrize(
    "partition_interval, partition_number, test_date, expected_partition_info, expected_partition_expression",
    [
        # Test cases with different partition intervals and partition numbers
        (
            "DAY",
            3,
            datetime(2024, 1, 1),
            [
                ("20240101", "20240102"),
                ("20240102", "20240103"),
                ("20240103", "20240104"),
            ],
            "DAY(activation_time)",
        ),
        (
            "MONTH",
            2,
            datetime(2024, 1, 1),
            [
                ("202401", "202402"),
                ("202402", "202403"),
            ],
            "MONTH(activation_time)",
        ),
        (
            "YEARWEEK",
            2,
            datetime(2024, 12, 31),
            [
                ("202501", "202502"),
                ("202502", "202503"),
            ],
            "YEARWEEK(activation_time, 1)",
        ),
    ],
)
def test_create_partitions(
    db: sqlalchemy.orm.Session,
    partition_interval,
    partition_number,
    test_date,
    expected_partition_info,
    expected_partition_expression,
):
    with (
        unittest.mock.patch(
            "services.api.utils.db.partitioner.datetime"
        ) as mock_datetime,
        unittest.mock.patch.object(
            framework.utils.singletons.db.get_db(),
            "create_partitions",
        ) as mocked_db_create_partitions,
    ):
        mock_datetime.now.return_value = test_date
        services.api.utils.db.partitioner.MySQLPartitioner().create_partitions(
            db,
            "alert_activations",
            partition_number,
            mlrun.common.schemas.partition.PartitionInterval(partition_interval),
        )

        # Verify that create_partitions was called with the expected partition information
        mocked_db_create_partitions.assert_called_once_with(
            session=db,
            table_name="alert_activations",
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
def test_get_interval(
    db: sqlalchemy.orm.Session,
    mocked_partition_expression,
    expected_partition_interval,
):
    with (
        unittest.mock.patch.object(
            framework.utils.singletons.db.get_db(),
            "get_partition_expression_for_table",
        ) as mocked_get_partition_expression_for_table,
    ):
        mocked_get_partition_expression_for_table.return_value = (
            mocked_partition_expression
        )
        partition_interval = (
            services.api.utils.db.partitioner.MySQLPartitioner().get_partition_interval(
                db,
                "alert_activations",
            )
        )
        assert partition_interval == expected_partition_interval
