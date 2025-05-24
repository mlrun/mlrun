# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add missing on cascade delete to tag,tag_v2,artifact_tag and notification tables

Revision ID: 528d9f5ea91d
Revises: 6925effc8fb1
Create Date: 2025-05-22 19:11:42.445053

"""

from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = "528d9f5ea91d"
down_revision = "6925effc8fb1"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column(
        "alert_configs_notifications",
        "parent_id",
        existing_type=mysql.INTEGER(),
        nullable=False,
    )
    op.drop_constraint(
        "alert_configs_notifications_ibfk_1",
        "alert_configs_notifications",
        type_="foreignkey",
    )
    op.create_foreign_key(
        None,
        "alert_configs_notifications",
        "alert_configs",
        ["parent_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.drop_constraint(
        "_artifacts_tags_obj_id_fk",
        "artifacts_tags",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "_artifacts_tags_obj_id_fk",
        "artifacts_tags",
        "artifacts",
        ["obj_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.alter_column(
        "runs_notifications",
        "parent_id",
        existing_type=mysql.INTEGER(),
        nullable=False,
    )
    op.drop_constraint(
        "runs_notifications_ibfk_1",
        "runs_notifications",
        type_="foreignkey",
    )
    op.create_foreign_key(
        None,
        "runs_notifications",
        "runs",
        ["parent_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.drop_constraint(
        "_runs_tags_obj_id_fk",
        "runs_tags",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "_runs_tags_obj_id_fk",
        "runs_tags",
        "runs",
        ["obj_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade():
    op.drop_constraint(
        None,
        "alert_configs_notifications",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "alert_configs_notifications_ibfk_1",
        "alert_configs_notifications",
        "alert_configs",
        ["parent_id"],
        ["id"],
    )
    op.alter_column(
        "alert_configs_notifications",
        "parent_id",
        existing_type=mysql.INTEGER(),
        nullable=True,
    )

    op.drop_constraint(
        "_artifacts_tags_obj_id_fk",
        "artifacts_tags",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "_artifacts_tags_obj_id_fk",
        "artifacts_tags",
        "artifacts",
        ["obj_id"],
        ["id"],
    )

    op.drop_constraint(
        None,
        "runs_notifications",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "runs_notifications_ibfk_1",
        "runs_notifications",
        "runs",
        ["parent_id"],
        ["id"],
    )
    op.alter_column(
        "runs_notifications",
        "parent_id",
        existing_type=mysql.INTEGER(),
        nullable=True,
    )

    op.drop_constraint(
        "_runs_tags_obj_id_fk",
        "runs_tags",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "_runs_tags_obj_id_fk",
        "runs_tags",
        "runs",
        ["obj_id"],
        ["id"],
    )
