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

"""Add partition key column for alerts activations partitioning in postgres"""

import logging
import os

import sqlalchemy
import sqlalchemy as sa
from alembic import op

import mlrun.common.db.dialects
import mlrun.common.schemas.partition

# revision identifiers, used by Alembic.
revision = "c25e56faecce"
down_revision = "6e8e4df16a4e"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade():
    # 1) add as NULLable so we can back‑fill
    op.add_column(
        "alert_activations",
        sa.Column("partition_key", sa.Integer(), nullable=True),
    )

    # 2) back‑fill on MySQL only
    bind = op.get_bind()
    if bind.dialect.name.startswith(mlrun.common.db.dialects.Dialects.MYSQL):
        interval = mlrun.common.schemas.partition.PartitionInterval(
            os.getenv("PARTITION_INTERVAL", "YEARWEEK").upper()
        )
        expression = interval.get_partition_expression(
            column_name="activation_time",
            dialect=mlrun.common.db.dialects.Dialects.MYSQL,
        )
        table = sqlalchemy.table(
            "alert_activations",
            sqlalchemy.column("activation_time", sqlalchemy.DateTime(timezone=True)),
            sqlalchemy.column("partition_key", sqlalchemy.Integer()),
        )
        op.execute(sqlalchemy.update(table).values(partition_key=expression))
    else:
        logger.info(
            "Skipping partition_key back‑fill: dialect '%s' is not MySQL",
            bind.dialect.name,
        )

    # 3) drop old PK and create new composite PK including partition_key
    op.drop_constraint("_alert_activation_uc", "alert_activations", type_="primary")
    op.create_primary_key(
        "_alert_activation_uc",
        "alert_activations",
        ["id", "activation_time", "partition_key"],
    )

    # 4) now make partition_key non‑nullable
    op.alter_column("alert_activations", "partition_key", nullable=False)


def downgrade():
    # 1) drop the composite PK
    op.drop_constraint("_alert_activation_uc", "alert_activations", type_="primary")
    # 2) restore original PK on (id, activation_time)
    op.create_primary_key(
        "_alert_activation_uc",
        "alert_activations",
        ["id", "activation_time"],
    )
    # 3) drop the column
    op.drop_column("alert_activations", "partition_key")
