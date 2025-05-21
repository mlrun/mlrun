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

"""add alert activation

Revision ID: 650f0ce2da6f
Revises: 63a7eec6d034
Create Date: 2024-10-30 16:38:07.592754
"""

import os
from datetime import datetime

import sqlalchemy as sa
from alembic import op

import mlrun.common.schemas.partition
from mlrun.db.sql_types import DateTime

# revision identifiers, used by Alembic.
revision = "650f0ce2da6f"
down_revision = "63a7eec6d034"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    dialect = bind.dialect.name
    # on Postgres, DECLARE RANGE partitioning up front
    table_kwargs = {}
    if dialect == "postgresql":
        table_kwargs["postgresql_partition_by"] = "RANGE (activation_time)"

    op.create_table(
        "alert_activations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("activation_time", DateTime(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("project", sa.String(255), nullable=False),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column("entity_id", sa.String(255), nullable=False),
        sa.Column("entity_kind", sa.String(255), nullable=False),
        sa.Column("event_kind", sa.String(255), nullable=False),
        sa.Column("severity", sa.String(255), nullable=False),
        sa.Column("number_of_events", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("activation_time", "id", name="_alert_activation_uc"),
        **table_kwargs,
    )
    op.create_index(
        "ix_alert_activation_activation_time",
        "alert_activations",
        ["activation_time"],
        unique=False,
    )
    op.create_index(
        "ix_alert_activation_project_name",
        "alert_activations",
        ["project", "name"],
        unique=False,
    )

    # only MySQL supports this ALTER … PARTITION BY syntax
    if dialect == "mysql":
        interval_name = os.getenv("PARTITION_INTERVAL", "YEARWEEK").upper()
        if not mlrun.common.schemas.partition.PartitionInterval.is_valid(interval_name):
            raise ValueError(
                f"Partition interval must be one of: "
                f"{mlrun.common.schemas.partition.PartitionInterval.valid_intervals()}"
            )
        now_utc = datetime.utcnow()
        interval = mlrun.common.schemas.partition.PartitionInterval(interval_name)
        pname, pval = interval.get_partition_info(now_utc)[0]
        expr = interval.get_partition_expression(column_name="activation_time")
        op.execute(f"""
            ALTER TABLE alert_activations
            PARTITION BY RANGE ({expr}) (
              PARTITION p{pname} VALUES LESS THAN ({pval})
            );
        """)


def downgrade():
    op.drop_table("alert_activations")
