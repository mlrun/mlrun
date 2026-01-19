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
import sqlalchemy as sa
from alembic import op

import mlrun.common.schemas.partition_interval

import framework.db.sqldb.sql_types

"""
Migration: add partition_key column to alert_activations, populate it, and update
the primary key to include it. Also persist the partition interval configuration
for alert_activations in table_partition_interval.
"""

revision = "c25e56faecce"
down_revision = "6d1d53f60e90"


def _update_partition_keys_bulk(
    connection: sa.engine.Connection,
    partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
) -> None:
    partition_expression = partition_interval.get_mysql_partition_key_sql(
        column_name="activation_time",
    )
    sql = f"""
        UPDATE alert_activations
        SET partition_key = {partition_expression}
    """
    connection.execute(sa.text(sql))


def upgrade() -> None:
    op.create_table(
        "table_partition_interval",
        sa.Column(
            "table_name",
            framework.db.sqldb.sql_types.Utf8BinText(),
            nullable=False,
        ),
        sa.Column(
            "interval",
            sa.Enum(
                "DAY",
                "MONTH",
                "YEARWEEK",
                name="partition_interval",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("table_name"),
    )

    connection = op.get_bind()
    is_mysql = connection.dialect.name == "mysql"

    # NOTE: This is the last place where PARTITION_INTERVAL is read from the
    # environment. From this point onward, the partition interval configuration
    # is persisted in the `table_partition_interval` table and all runtime and
    # migration logic must read it from there instead of the environment.
    partition_interval = mlrun.common.schemas.partition_interval.PartitionInterval.get_partition_interval_from_env()

    # Save configured interval for this table
    table_partition_interval = sa.table(
        "table_partition_interval",
        sa.column("table_name"),
        sa.column("interval"),
    )
    connection.execute(
        table_partition_interval.insert().values(
            table_name="alert_activations",
            interval=partition_interval.name,
        )
    )

    # This migration is relevant only for MySQL.
    # Newer PostgreSQL-based installations create the schema directly via
    # SQLAlchemy (including partitioning-related columns), and therefore do not
    # require any backfill or primary key alteration at migration time.
    if not is_mysql:
        return

    # 1. Add column nullable
    op.add_column(
        "alert_activations",
        sa.Column("partition_key", sa.Integer(), nullable=True),
    )

    # 2. Backfill values
    _update_partition_keys_bulk(
        connection=connection,
        partition_interval=partition_interval,
    )

    # 3. Make NOT NULL
    op.alter_column(
        "alert_activations",
        "partition_key",
        existing_type=sa.Integer(),
        nullable=False,
    )

    # 4. Replace PK in a *single* ALTER TABLE so MySQL does not complain
    #    about leaving the AUTO_INCREMENT column without a key.
    op.execute(
        """
        ALTER TABLE alert_activations
        DROP PRIMARY KEY,
        ADD PRIMARY KEY (id, activation_time, partition_key)
        """
    )


def downgrade() -> None:
    connection = op.get_bind()
    is_mysql = connection.dialect.name == "mysql"

    if is_mysql:
        # Reverse PK change, also as a single ALTER TABLE
        op.execute(
            """
            ALTER TABLE alert_activations
            DROP PRIMARY KEY,
            ADD PRIMARY KEY (id, activation_time)
            """
        )
        op.drop_column("alert_activations", "partition_key")

    op.drop_table("table_partition_interval")
