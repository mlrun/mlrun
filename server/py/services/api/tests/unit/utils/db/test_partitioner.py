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
import unittest.mock
from datetime import datetime

import pytest

import mlrun.common.schemas.partition_interval

import framework.db.sqldb.partition_bootstrapper
import framework.utils.singletons.db
import services.api.utils.db.partitioner as part_mod


@pytest.mark.parametrize(
    "interval, dt, exp_name, exp_next_val",
    [
        ("DAY", datetime(2024, 10, 30), "p20241030", 20241031),
        ("MONTH", datetime(2024, 10, 30), "p202410", 202411),
        ("YEARWEEK", datetime(2024, 10, 30), "p202444", 202445),
        ("YEARWEEK", datetime(2023, 1, 1), "p202252", 202301),
        ("YEARWEEK", datetime(2024, 12, 31), "p202501", 202502),
        ("YEARWEEK", datetime(2024, 1, 1), "p202401", 202402),
        ("YEARWEEK", datetime(2024, 6, 15), "p202424", 202425),
    ],
)
def test_get_partition_info_for_datetime(interval, dt, exp_name, exp_next_val):
    info = mlrun.common.schemas.partition_interval.PartitionInterval(
        interval
    ).get_partition_names_and_boundaries(dt)[0]
    assert info == (exp_name, exp_next_val)


@pytest.mark.parametrize(
    "interval, retention_days, now_dt, exp_cutoff",
    [
        ("DAY", 4 * 7, datetime(2024, 1, 1), "p20231204"),
        ("DAY", 1, datetime(2024, 1, 1), "p20231231"),
        ("MONTH", 6 * 7, datetime(2024, 7, 15), "p202406"),
        ("YEARWEEK", 12 * 7, datetime(2024, 6, 1), "p202410"),
        ("YEARWEEK", 14 * 7, datetime(2024, 6, 1), "p202408"),
    ],
)
def test_drop_partitions(db, interval, retention_days, now_dt, exp_cutoff):
    with (
        unittest.mock.patch(f"{part_mod.__name__}.datetime") as mock_dt,
        unittest.mock.patch.object(
            framework.utils.singletons.db.get_db(), "drop_partitions"
        ) as mock_drop,
    ):
        mock_dt.now.return_value = now_dt
        part_mod.DBPartitioner().drop_partitions(
            session=db,
            table_name="alert_activations",
            partition_interval=mlrun.common.schemas.partition_interval.PartitionInterval(
                interval
            ),
            retention_days=retention_days,
        )
        mock_drop.assert_called_once_with(
            session=db,
            table_name="alert_activations",
            cutoff_partition_name=exp_cutoff,
        )


@pytest.mark.parametrize(
    "interval, partitions_to_create, now_dt",
    [
        (
            mlrun.common.schemas.partition_interval.PartitionInterval.DAY,
            3,
            datetime(2024, 1, 1),
        ),
        (
            mlrun.common.schemas.partition_interval.PartitionInterval.MONTH,
            2,
            datetime(2024, 1, 1),
        ),
        (
            mlrun.common.schemas.partition_interval.PartitionInterval.YEARWEEK,
            2,
            datetime(2024, 12, 31),
        ),
    ],
)
def test_create_partitions(db, interval, partitions_to_create, now_dt):
    with (
        unittest.mock.patch(f"{part_mod.__name__}.datetime") as mock_dt,
        unittest.mock.patch.object(
            framework.db.sqldb.partition_bootstrapper.PartitionBootstrapperSqlite,
            "bootstrap",
        ) as mock_bootstrap,
    ):
        mock_dt.now.return_value = now_dt
        # Ensure no calls leaked from metadata bootstrap
        mock_bootstrap.reset_mock()

        buffer_multiplier_override = 0  # override so partition_count == retain
        part_mod.DBPartitioner(
            buffer_multiplier_override=buffer_multiplier_override,
        ).create_partitions(
            session=db,
            table_name="alert_activations",
            partitions_to_create=partitions_to_create,
            partition_interval=interval,
        )

        mock_bootstrap.assert_called_once_with(
            session=db,
            table_name="alert_activations",
            partition_interval=interval,
            partitions_count=partitions_to_create,
        )
