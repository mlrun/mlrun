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

"""Add background task labels

Revision ID: d8f7c6b5a4e3
Revises: b31651280cce
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d8f7c6b5a4e3"
down_revision = "b31651280cce"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "background_task_labels",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "task_id",
            sa.Integer,
            sa.ForeignKey("background_tasks.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255, collation="utf8mb3_bin"), nullable=False),
        sa.Column("value", sa.String(255, collation="utf8mb3_bin"), nullable=True),
        sa.UniqueConstraint(
            "task_id", "name", name="uq_bg_task_labels_task_id_and_name"
        ),
    )
    op.create_index(
        "ix_background_tasks_state",
        "background_tasks",
        ["state"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_background_tasks_state", table_name="background_tasks")
    op.drop_index(
        "ix_background_task_labels_task_id", table_name="background_task_labels"
    )
    op.drop_table("background_task_labels")
