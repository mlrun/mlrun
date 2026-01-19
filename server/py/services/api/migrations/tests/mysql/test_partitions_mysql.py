# Copyright 2025 Iguazio
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
from datetime import datetime

import pytest
import sqlalchemy
import sqlalchemy.orm.session

import mlrun.common.schemas.partition_interval
import tests.common_fixtures
import tests.conftest

import framework.db.sqldb.db
import framework.db.sqldb.db as sqldb
import framework.db.sqldb.partition_bootstrapper
import services.api.utils.db.partitioner


@pytest.mark.integration
@pytest.mark.parametrize(
    "date_str",
    [
        "2007-12-31",
        "2008-01-01",
        "2015-12-31",
        "2016-01-04",
        "2019-12-30",
        "2020-01-01",
        "2020-12-28",
        "2021-01-04",
        "2024-12-31",
        "2020-06-15",
        "2021-07-10",
    ],
)
def test_partition_interval_yearweek_matches_mysql_yearweek_mode_1(
    mysql_db_session: sqlalchemy.orm.session.Session,
    date_str: str,
):
    """
    Validate that PartitionInterval.YEARWEEK.get_partition_key_value matches
    MySQL YEARWEEK(date, 1) for selected dates, especially around year boundaries.
    """
    date_time = datetime.fromisoformat(date_str)
    python_value = mlrun.common.schemas.partition_interval.PartitionInterval.YEARWEEK.get_partition_key_value(
        date_time
    )

    mysql_value = mysql_db_session.execute(
        sqlalchemy.text("SELECT YEARWEEK(:date_str, 1)"),
        {"date_str": date_str},
    ).scalar_one()

    assert (
        python_value == mysql_value
    ), f"{date_str}: python={python_value}, mysql={mysql_value}"


@pytest.mark.integration
@tests.common_fixtures.freeze_datetime(datetime(2025, 1, 1))
def test_create_partitions_mysql(
    mysql_db_session: sqlalchemy.orm.session.Session,
):
    table = "dyn_table"

    mysql_db_session.execute(
        sqlalchemy.text(
            f"""
        CREATE TABLE `{table}` (
            id            INT NOT NULL,
            partition_key INT NOT NULL,
            data          TEXT
        ) PARTITION BY RANGE (partition_key) (
            PARTITION p0 VALUES LESS THAN (1)
        );
    """
        )
    )

    initial_partition_names = set(
        framework.db.sqldb.db.MySQLDB._get_partition_metadata(
            session=mysql_db_session,
            table_name=table,
        ).keys()
    )

    services.api.utils.db.partitioner.DBPartitioner(
        buffer_multiplier_override=0
    ).create_partitions(
        session=mysql_db_session,
        table_name=table,
        partitions_to_create=2,
        partition_interval=mlrun.common.schemas.partition_interval.PartitionInterval.DAY,
    )

    day_interval = mlrun.common.schemas.partition_interval.PartitionInterval.DAY
    expected_new_partition_names = {
        day_interval.get_partition_name(datetime(2025, 1, 1)),
        day_interval.get_partition_name(datetime(2025, 1, 2)),
    }

    actual_partition_names = set(
        framework.db.sqldb.db.MySQLDB._get_partition_metadata(
            session=mysql_db_session,
            table_name=table,
        ).keys()
    )

    # all original partitions must remain
    assert initial_partition_names <= actual_partition_names

    # all expected new partitions must exist
    assert expected_new_partition_names <= actual_partition_names

    # no unexpected partitions
    assert (
        actual_partition_names == initial_partition_names | expected_new_partition_names
    )

    mysql_db_session.close()


@pytest.mark.integration
@tests.common_fixtures.freeze_datetime(datetime(2025, 1, 6))
def test_drop_partitions_mysql(
    mysql_db_session: sqlalchemy.orm.session.Session,
):
    table = "dyn_table_drop"

    # skeleton table (NOT partitioned yet, but PK includes partition_key)
    mysql_db_session.execute(
        sqlalchemy.text(
            f"""
            CREATE TABLE `{table}` (
                id            INT NOT NULL,
                partition_key INT NOT NULL,
                data          TEXT,
                PRIMARY KEY (id, partition_key)
            );
        """
        )
    )

    # bootstrap 3 weekly partitions starting at 2025‑01‑06
    framework.db.sqldb.partition_bootstrapper.PartitionBootstrapper(
        mysql_db_session.get_bind().dialect.name
    ).bootstrap(
        session=mysql_db_session,
        table_name=table,
        partition_interval=mlrun.common.schemas.partition_interval.PartitionInterval.YEARWEEK,
        partitions_count=3,
    )

    part_info = mlrun.common.schemas.partition_interval.PartitionInterval.YEARWEEK.get_partition_names_and_boundaries(
        start_datetime=datetime(2025, 1, 6),  # frozen “now”
        partitions_count=3,
    )

    # advance time two weeks and purge anything older than 7 days
    tests.common_fixtures.FrozenDatetime._frozen_now = datetime(2025, 1, 20)
    services.api.utils.db.partitioner.DBPartitioner(
        buffer_multiplier_override=0
    ).drop_partitions(
        session=mysql_db_session,
        table_name=table,
        partition_interval=mlrun.common.schemas.partition_interval.PartitionInterval.YEARWEEK,
        retention_days=7,
    )

    cutoff_name = part_info[1][0]  # should survive
    remaining = set(
        sqldb.MySQLDB._get_partition_metadata(mysql_db_session, table).keys()
    )

    # oldest partition gone, newer ones kept
    assert cutoff_name in remaining
    assert part_info[0][0] not in remaining
    assert {n for n, _ in part_info[2:]} <= remaining

    mysql_db_session.close()
