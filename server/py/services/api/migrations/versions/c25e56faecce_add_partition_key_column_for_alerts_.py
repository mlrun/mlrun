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

revision = "c25e56faecce"
down_revision = "6d1d53f60e90"

TABLE_NAME = "alert_activations"
_PK_RAW = "pk_alert_activations"


def _fill_partition_keys(
    connection, interval: mlrun.common.schemas.partition_interval.PartitionInterval
) -> None:
    """Populate partition_key for all existing rows."""
    metadata = sa.MetaData()
    alert_table = sa.Table(
        TABLE_NAME,
        metadata,
        autoload_with=connection,
    )

    select_stmt = sa.select(alert_table.c.id, alert_table.c.activation_time)
    for row_id, act_time in connection.execute(select_stmt):
        connection.execute(
            alert_table.update()
            .where(
                (alert_table.c.id == row_id)
                & (alert_table.c.activation_time == act_time)
            )
            .values(partition_key=interval.get_partition_key_value(act_time))
        )


def _swap_pk_mysql(connection):
    connection.execute(
        sa.text(
            f"""
        ALTER TABLE {TABLE_NAME}
        DROP PRIMARY KEY,
        ADD PRIMARY KEY (id, activation_time, partition_key)
        """
        )
    )


def upgrade() -> None:
    op.create_table(
        "table_partition_interval",
        sa.Column(
            "table_name", framework.db.sqldb.sql_types.Utf8BinText(), nullable=False
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
    engine = op.get_bind()
    interval = mlrun.common.schemas.partition_interval.PartitionInterval.get_partition_interval_from_env()

    op.add_column(TABLE_NAME, sa.Column("partition_key", sa.Integer(), nullable=True))
    _fill_partition_keys(engine, interval)
    op.alter_column(
        TABLE_NAME, "partition_key", existing_type=sa.Integer(), nullable=False
    )

    pk_name = op.f("pk_alert_activations")

    if engine.dialect.name == "mysql":
        # MySQL: one‑shot ALTER avoids auto‑increment error
        _swap_pk_mysql(engine)
    else:
        # Postgres etc. can drop + create in two steps
        op.drop_constraint(pk_name, TABLE_NAME, type_="primary")
        op.create_primary_key(
            pk_name, TABLE_NAME, ["id", "activation_time", "partition_key"]
        )


def _swap_pk_mysql_back(connection):
    connection.execute(
        sa.text(
            f"""
            ALTER TABLE {TABLE_NAME}
            DROP PRIMARY KEY,
            ADD PRIMARY KEY (id, activation_time)
            """
        )
    )


def downgrade() -> None:
    engine = op.get_bind()
    pk_name = op.f(_PK_RAW)

    if engine.dialect.name == "mysql":
        # MySQL must keep the AUTO_INCREMENT column keyed at all times
        _swap_pk_mysql_back(engine)
    else:
        op.drop_constraint(pk_name, TABLE_NAME, type_="primary")
        op.create_primary_key(pk_name, TABLE_NAME, ["id", "activation_time"])

    # column is no longer part of the PK – safe to remove now
    op.drop_column(TABLE_NAME, "partition_key")

    # clean up the helper table introduced in upgrade()
    op.drop_table("table_partition_interval")
