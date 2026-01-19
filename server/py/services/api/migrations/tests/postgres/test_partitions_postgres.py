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
import sqlalchemy.orm

import mlrun.common.schemas.partition_interval
import tests.common_fixtures

import framework.db.sqldb.db as sqldb
import framework.db.sqldb.partition_bootstrapper
import services.api.utils.db.partitioner

pytest.importorskip("psycopg2", reason="psycopg2 not installed")


@pytest.mark.integration
@pytest.mark.usefixtures("pmr_postgres_container")
@tests.common_fixtures.freeze_datetime(datetime(2025, 1, 1))
def test_create_partitions_postgres(
    postgres_db_session: sqlalchemy.orm.session.Session,
):
    table = "dyn_table"

    # base table – no partitions yet
    postgres_db_session.execute(
        sqlalchemy.text(
            f"""
            CREATE TABLE {table} (
                id            INTEGER NOT NULL,
                partition_key INTEGER NOT NULL,
                data          TEXT,
                PRIMARY KEY (id, partition_key)
            ) PARTITION BY RANGE (partition_key);
            """
        )
    )

    # bootstrap two daily partitions starting at frozen 2025‑01‑01
    framework.db.sqldb.partition_bootstrapper.PartitionBootstrapper(
        postgres_db_session.get_bind().dialect.name
    ).bootstrap(
        session=postgres_db_session,
        table_name=table,
        partition_interval=mlrun.common.schemas.partition_interval.PartitionInterval.DAY,
        partitions_count=2,
    )

    expected = {
        n
        for n, _ in mlrun.common.schemas.partition_interval.PartitionInterval.DAY.get_partition_names_and_boundaries(
            start_datetime=datetime(2025, 1, 1), partitions_count=2
        )
    }
    attached = set(
        sqldb.PostgreSQLDB._get_partition_metadata(postgres_db_session, table).keys()
    )
    assert attached == expected
    postgres_db_session.close()


@pytest.mark.integration
@pytest.mark.usefixtures("pmr_postgres_container")
@tests.common_fixtures.freeze_datetime(datetime(2025, 1, 6))
def test_drop_partitions_postgres(postgres_db_session):
    table = "dyn_table_drop"

    # base table – no partitions yet
    postgres_db_session.execute(
        sqlalchemy.text(
            f"""
            CREATE TABLE {table} (
                id            INTEGER NOT NULL,
                partition_key INTEGER NOT NULL,
                data          TEXT,
                PRIMARY KEY (id, partition_key)
            ) PARTITION BY RANGE (partition_key);
            """
        )
    )

    # 1. bootstrap two daily partitions for 2025‑01‑06 and 07
    framework.db.sqldb.partition_bootstrapper.PartitionBootstrapper(
        postgres_db_session.get_bind().dialect.name
    ).bootstrap(
        session=postgres_db_session,
        table_name=table,
        partition_interval=mlrun.common.schemas.partition_interval.PartitionInterval.DAY,
        partitions_count=2,
    )
    parts = mlrun.common.schemas.partition_interval.PartitionInterval.DAY.get_partition_names_and_boundaries(
        start_datetime=datetime(2025, 1, 6), partitions_count=2
    )

    # 2. advance clock by two days and drop anything older than 1 day
    tests.common_fixtures.FrozenDatetime._frozen_now = datetime(2025, 1, 8)
    services.api.utils.db.partitioner.DBPartitioner().drop_partitions(
        session=postgres_db_session,
        table_name=table,
        partition_interval=mlrun.common.schemas.partition_interval.PartitionInterval.DAY,
        retention_days=1,
    )

    cutoff_name = parts[1][0]  # should remain
    remaining = set(
        sqldb.PostgreSQLDB._get_partition_metadata(postgres_db_session, table).keys()
    )

    assert parts[0][0] not in remaining  # oldest dropped
    assert cutoff_name in remaining  # cutoff kept
    postgres_db_session.close()
