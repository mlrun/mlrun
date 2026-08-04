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


def _create_partitioned_table(
    session: sqlalchemy.orm.session.Session, table: str
) -> None:
    session.execute(
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


def _partition_bound_expr(
    session: sqlalchemy.orm.session.Session, partition_name: str
) -> str:
    return session.execute(
        sqlalchemy.text(
            "SELECT pg_get_expr(c.relpartbound, c.oid) "
            "FROM pg_class c WHERE c.relname = :name"
        ),
        {"name": partition_name},
    ).scalar_one()


def _bootstrap(
    session: sqlalchemy.orm.session.Session, table: str, partitions_count: int
) -> None:
    framework.db.sqldb.partition_bootstrapper.PartitionBootstrapper(
        session.get_bind().dialect.name
    ).bootstrap(
        session=session,
        table_name=table,
        partition_interval=mlrun.common.schemas.partition_interval.PartitionInterval.DAY,
        partitions_count=partitions_count,
    )


@pytest.mark.integration
@pytest.mark.usefixtures("pmr_postgres_container")
@tests.common_fixtures.freeze_datetime(datetime(2025, 6, 1))
def test_bootstrap_after_gap_does_not_overlap(
    postgres_db_session: sqlalchemy.orm.session.Session,
):
    """
    Regression for ORIS-3709: when the periodic task misses days, the next run's
    first new partition must anchor on the existing max bound (spanning the gap),
    not on MINVALUE. Pre-fix this raised
    ``psycopg2.errors.InvalidObjectDefinition: partition ... would overlap ...``.
    """
    table = "dyn_table_gap"
    _create_partitioned_table(postgres_db_session, table)

    # Day 0: single partition p20250601 -> [MINVALUE, 20250602).
    _bootstrap(postgres_db_session, table, partitions_count=1)

    # Skip days 2, 3, 4 entirely, then run again on day 5 (the missed-rotation gap).
    tests.common_fixtures.FrozenDatetime._frozen_now = datetime(2025, 6, 5)
    _bootstrap(postgres_db_session, table, partitions_count=2)  # must not raise

    metadata = sqldb.PostgreSQLDB._get_partition_metadata(postgres_db_session, table)
    assert metadata == {
        "p20250601": 20250602,
        "p20250605": 20250606,
        "p20250606": 20250607,
    }

    # The gap-filler is contiguous with the existing partition (not MINVALUE),
    # so it spans the skipped days without overlapping p20250601.
    gap_bound = _partition_bound_expr(postgres_db_session, "p20250605")
    assert "20250602" in gap_bound
    assert "MINVALUE" not in gap_bound

    # A row dated on a skipped gap day (2025-06-03) still has a home -> no hole.
    postgres_db_session.execute(
        sqlalchemy.text(
            f"INSERT INTO {table} (id, partition_key, data) "
            "VALUES (1, 20250603, 'gap-day')"
        )
    )
    postgres_db_session.commit()
    postgres_db_session.close()


@pytest.mark.integration
@pytest.mark.usefixtures("pmr_postgres_container")
@tests.common_fixtures.freeze_datetime(datetime(2025, 7, 1))
def test_bootstrap_contiguous_runs_are_idempotent(
    postgres_db_session: sqlalchemy.orm.session.Session,
):
    """Repeated runs create only missing partitions; a re-run adds nothing."""
    table = "dyn_table_contig"
    _create_partitioned_table(postgres_db_session, table)

    _bootstrap(postgres_db_session, table, partitions_count=2)
    first = set(
        sqldb.PostgreSQLDB._get_partition_metadata(postgres_db_session, table).keys()
    )
    assert first == {"p20250701", "p20250702"}

    # Same "now": nothing new, no error.
    _bootstrap(postgres_db_session, table, partitions_count=2)
    assert (
        set(
            sqldb.PostgreSQLDB._get_partition_metadata(
                postgres_db_session, table
            ).keys()
        )
        == first
    )

    # One day later: exactly one new contiguous partition is added.
    tests.common_fixtures.FrozenDatetime._frozen_now = datetime(2025, 7, 2)
    _bootstrap(postgres_db_session, table, partitions_count=2)
    assert set(
        sqldb.PostgreSQLDB._get_partition_metadata(postgres_db_session, table).keys()
    ) == {"p20250701", "p20250702", "p20250703"}
    postgres_db_session.close()


@pytest.mark.integration
@pytest.mark.usefixtures("pmr_postgres_container")
@tests.common_fixtures.freeze_datetime(datetime(2025, 8, 1))
def test_bootstrap_initial_uses_minvalue(
    postgres_db_session: sqlalchemy.orm.session.Session,
):
    """The very first partition of an empty table is anchored at MINVALUE."""
    table = "dyn_table_initial"
    _create_partitioned_table(postgres_db_session, table)

    _bootstrap(postgres_db_session, table, partitions_count=2)

    first_bound = _partition_bound_expr(postgres_db_session, "p20250801")
    assert "MINVALUE" in first_bound
    assert "20250802" in first_bound
    # The second partition chains from the first, not from MINVALUE.
    second_bound = _partition_bound_expr(postgres_db_session, "p20250802")
    assert "MINVALUE" not in second_bound
    assert "20250802" in second_bound
    postgres_db_session.close()


@pytest.mark.integration
@pytest.mark.usefixtures("pmr_postgres_container")
@tests.common_fixtures.freeze_datetime(datetime(2025, 9, 1))
def test_rotation_after_gap_creates_and_drops(
    postgres_db_session: sqlalchemy.orm.session.Session,
):
    """
    End-to-end ORIS-3709: after a multi-day gap, a rotation both creates the new
    partition (no overlap) and drops the expired one. Pre-fix the create raised,
    so neither happened.
    """
    table = "dyn_table_rotate"
    _create_partitioned_table(postgres_db_session, table)

    # Day 0 partition, then a 4-day gap.
    _bootstrap(postgres_db_session, table, partitions_count=1)
    tests.common_fixtures.FrozenDatetime._frozen_now = datetime(2025, 9, 5)

    # Create (spans the gap) then drop anything older than 1 day.
    _bootstrap(postgres_db_session, table, partitions_count=2)
    services.api.utils.db.partitioner.DBPartitioner().drop_partitions(
        session=postgres_db_session,
        table_name=table,
        partition_interval=mlrun.common.schemas.partition_interval.PartitionInterval.DAY,
        retention_days=1,
    )

    remaining = set(
        sqldb.PostgreSQLDB._get_partition_metadata(postgres_db_session, table).keys()
    )
    assert "p20250901" not in remaining  # expired, dropped
    assert {"p20250905", "p20250906"} <= remaining  # newly created, kept
    postgres_db_session.close()
