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
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

import mlrun.common.schemas as schemas

import framework.db.sqldb.db
import services.api.migrations.tests.base.conftest
import services.api.utils.db.partitioner


@pytest.mark.integration
@pytest.mark.usefixtures("pmr_mysql_container")
@services.api.migrations.tests.base.conftest.freeze_datetime(datetime(2025, 1, 1))
def test_create_partitions_mysql(alembic_engine):
    session = sessionmaker(bind=alembic_engine)()
    table = "dyn_table"

    session.execute(
        text(
            f"""
            CREATE TABLE `{table}` (
                id   INT NOT NULL,
                data TEXT
            ) PARTITION BY RANGE (id) (
                PARTITION p0 VALUES LESS THAN (1)
            );
            """
        )
    )

    partitioner = services.api.utils.db.partitioner.DBPartitioner()
    partitioner.create_partitions(
        session=session,
        table_name=table,
        partition_number=2,
        partition_interval=schemas.PartitionInterval.DAY,
    )

    expected_names = {
        name
        for name, _ in schemas.PartitionInterval.DAY.get_partition_info(
            datetime(2025, 1, 1),
            partition_number=2,
        )
    }
    expected_names.add("p0")

    actual_names = set(
        framework.db.sqldb.db.MySQLDB._get_partition_metadata(session, table).keys()
    )
    assert expected_names == actual_names
    session.close()


@pytest.mark.usefixtures("pmr_mysql_container")
@services.api.migrations.tests.base.conftest.freeze_datetime(datetime(2025, 1, 6))
def test_drop_partitions_mysql(alembic_engine):
    session = sessionmaker(bind=alembic_engine)()
    table = "dyn_table_drop"

    session.execute(
        text(
            f"""
            CREATE TABLE `{table}` (
                id   INT NOT NULL,
                data TEXT
            ) PARTITION BY RANGE (id) (
                PARTITION p0 VALUES LESS THAN (1)
            );
            """
        )
    )

    partitioner = services.api.utils.db.partitioner.DBPartitioner()
    partitioner.create_partitions(
        session=session,
        table_name=table,
        partition_number=3,
        partition_interval=schemas.PartitionInterval.YEARWEEK,
    )

    parts = schemas.PartitionInterval.YEARWEEK.get_partition_info(
        datetime(2025, 1, 6),
        partition_number=3,
    )

    # advance time two weeks before dropping
    services.api.migrations.tests.base.conftest.FrozenDatetime._frozen_now = datetime(
        2025, 1, 20
    )

    partitioner.drop_partitions(
        session=session,
        table_name=table,
        retention_days=7,
        partition_interval=schemas.PartitionInterval.YEARWEEK,
    )

    cutoff_name = parts[1][0]

    remaining = set(
        framework.db.sqldb.db.MySQLDB._get_partition_metadata(session, table).keys()
    )
    assert cutoff_name in remaining
    assert parts[0][0] not in remaining
    newer = {name for name, _ in parts[2:]}
    assert newer <= remaining

    session.close()
