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

PRIMARY_KEY_NAME = "_alert_activation_uc"


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

    partition_interval = mlrun.common.schemas.partition_interval.PartitionInterval.get_partition_interval_from_env()

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

    # On non-MySQL dialects we only persist the interval; no schema / data changes
    if not is_mysql:
        return

    # MySQL-specific: add partition_key nullable, backfill, and then enforce NOT NULL + new PK
    op.add_column(
        "alert_activations",
        sa.Column("partition_key", sa.Integer(), nullable=True),
    )

    _update_partition_keys_bulk(
        connection=connection,
        partition_interval=partition_interval,
    )

    op.alter_column(
        "alert_activations",
        "partition_key",
        existing_type=sa.Integer(),
        nullable=False,
    )

    op.drop_constraint(
        PRIMARY_KEY_NAME,
        "alert_activations",
        type_="primary",
    )
    op.create_primary_key(
        PRIMARY_KEY_NAME,
        "alert_activations",
        ["id", "activation_time", "partition_key"],
    )


def downgrade() -> None:
    engine = op.get_bind()
    is_mysql = engine.dialect.name == "mysql"

    if is_mysql:
        op.drop_constraint(
            PRIMARY_KEY_NAME,
            "alert_activations",
            type_="primary",
        )
        op.create_primary_key(
            PRIMARY_KEY_NAME,
            "alert_activations",
            ["id", "activation_time"],
        )
        op.drop_column("alert_activations", "partition_key")

    op.drop_table("table_partition_interval")
