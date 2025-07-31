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

import mlrun.common.schemas

import framework.db.sqldb.db

pytest.importorskip(
    "psycopg2",
    reason="psycopg2 not installed",
)


@pytest.mark.integration
def test_create_partitions_postgres(alembic_engine):
    session = sessionmaker(bind=alembic_engine)()
    table = "dyn_table"

    session.execute(
        text(f"""
        CREATE TABLE {table} (
            id   INTEGER NOT NULL,
            data TEXT
        ) PARTITION BY RANGE (id);

        CREATE TABLE {table}_p0 PARTITION OF {table}
        FOR VALUES FROM (MINVALUE) TO (1);
        """)
    )

    start = datetime(2025, 1, 1)
    parts = mlrun.common.schemas.PartitionInterval.DAY.get_partition_info(
        start, partition_number=2
    )
    framework.db.sqldb.db.PostgreSQLDB.create_partitions(session, table, parts)

    attached = set(
        framework.db.sqldb.db.PostgreSQLDB._get_partition_metadata(
            session, table
        ).keys()
    )
    expected = {name for name, _ in parts}.union({f"{table}_p0"})
    assert attached == expected
    session.close()


@pytest.mark.integration
def test_drop_partitions_postgres(alembic_engine):
    session = sessionmaker(bind=alembic_engine)()
    table = "dyn_table_drop"

    # 1) base table + seed p0
    session.execute(
        text(f"""
        CREATE TABLE {table} (
            id   INTEGER NOT NULL,
            data TEXT
        ) PARTITION BY RANGE (id);

        CREATE TABLE {table}_p0 PARTITION OF {table}
        FOR VALUES FROM (MINVALUE) TO (1);
        """)
    )

    start = datetime(2025, 1, 6)
    parts = mlrun.common.schemas.PartitionInterval.YEARWEEK.get_partition_info(
        start, partition_number=3
    )
    framework.db.sqldb.db.PostgreSQLDB.create_partitions(session, table, parts)

    cutoff = parts[1][0]
    framework.db.sqldb.db.PostgreSQLDB.drop_partitions(
        session, table, cutoff_partition_name=cutoff
    )

    remaining = set(
        framework.db.sqldb.db.PostgreSQLDB._get_partition_metadata(
            session, table
        ).keys()
    )

    assert parts[0][0] not in remaining  # oldest gone
    assert cutoff in remaining  # cutoff kept
    newer = {name for name, _ in parts[2:]}  # newest kept
    assert newer <= remaining
    session.close()
