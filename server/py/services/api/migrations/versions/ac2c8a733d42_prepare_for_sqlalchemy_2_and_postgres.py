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

"""Prepare for SQLAlchemy 2 and Postgres

Revision ID: ac2c8a733d42
Revises: 6925effc8fb1
Create Date: 2025-05-13 20:02:34.836350
"""

from alembic import op
from sqlalchemy import inspect

import mlrun.db.sql_types

# revision identifiers, used by Alembic.
revision = "ac2c8a733d42"
down_revision = "6925effc8fb1"
branch_labels = None
depends_on = None


def upgrade():
    # alter timestamp columns to MicroSecondDateTime
    tables = {
        "alert_configs_notifications": ["sent_time"],
        "alert_states": ["created", "last_updated"],
        "artifacts": ["updated"],
        "artifacts_v2": ["created", "updated"],
        "background_tasks": ["created", "updated"],
        "data_versions": ["created"],
        "feature_sets": ["created", "updated"],
        "feature_vectors": ["created", "updated"],
        "functions": ["updated"],
        "hub_sources": ["created", "updated"],
        "model_endpoints": ["created", "updated"],
        "pagination_cache": ["last_accessed"],
        "projects": ["created"],
        "runs": ["start_time", "updated"],
        "runs_notifications": ["sent_time"],
        "schedules_v2": ["creation_time", "next_run_time"],
    }
    for tbl, cols in tables.items():
        for col in cols:
            op.alter_column(
                tbl,
                col,
                existing_type=mlrun.db.sql_types.DateTime(),
                type_=mlrun.db.sql_types.MicroSecondDateTime(),
                existing_nullable=True,
            )

    # drop legacy constraints and indexes
    conn = op.get_bind()
    dialect = conn.dialect.name
    inspector = inspect(conn)

    # legacy FKs on artifacts_v2 tables
    if "artifacts_v2_labels" in inspector.get_table_names():
        fk_names = [
            fk["name"] for fk in inspector.get_foreign_keys("artifacts_v2_labels")
        ]
        if "artifacts_v2_labels_parent_fkey" in fk_names:
            op.drop_constraint(
                "artifacts_v2_labels_parent_fkey",
                "artifacts_v2_labels",
                type_="foreignkey",
            )

    if "artifacts_v2_tags" in inspector.get_table_names():
        fk_names = [
            fk["name"] for fk in inspector.get_foreign_keys("artifacts_v2_tags")
        ]
        if "artifacts_v2_tags_obj_id_fkey" in fk_names:
            op.drop_constraint(
                "artifacts_v2_tags_obj_id_fkey", "artifacts_v2_tags", type_="foreignkey"
            )

    # legacy unique on name
    for tbl in ("feature_sets", "feature_vectors", "functions"):
        if tbl in inspector.get_table_names():
            if dialect == "postgresql":
                cons = inspector.get_unique_constraints(tbl)
                if any(c["name"] == f"{tbl}_name_key" for c in cons):
                    op.drop_constraint(f"{tbl}_name_key", tbl, type_="unique")
            else:
                idx = inspector.get_indexes(tbl)
                if any(i["name"] == f"{tbl}_name_key" for i in idx):
                    op.drop_index(f"{tbl}_name_key", table_name=tbl)


def downgrade():
    # recreate legacy constraints and revert types
    conn = op.get_bind()
    dialect = conn.dialect.name
    inspector = inspect(conn)

    # restore unique on name
    for tbl in ("feature_sets", "feature_vectors", "functions"):
        if tbl in inspector.get_table_names():
            if dialect == "postgresql":
                cons = inspector.get_unique_constraints(tbl)
                if not any(c["name"] == f"{tbl}_name_key" for c in cons):
                    op.create_unique_constraint(f"{tbl}_name_key", tbl, ["name"])
            else:
                idx = inspector.get_indexes(tbl)
                if not any(i["name"] == f"{tbl}_name_key" for i in idx):
                    op.create_index(f"{tbl}_name_key", tbl, ["name"], unique=True)

    # restore FKs on artifacts_v2 if the columns exist
    if "artifacts_v2_labels" in inspector.get_table_names():
        cols = [c["name"] for c in inspector.get_columns("artifacts_v2_labels")]
        fk_names = [
            fk["name"] for fk in inspector.get_foreign_keys("artifacts_v2_labels")
        ]
        if "parent_id" in cols and "artifacts_v2_labels_parent_fkey" not in fk_names:
            op.create_foreign_key(
                "artifacts_v2_labels_parent_fkey",
                "artifacts_v2_labels",
                "artifacts_v2",
                ["parent_id"],
                ["id"],
            )

    if "artifacts_v2_tags" in inspector.get_table_names():
        cols = [c["name"] for c in inspector.get_columns("artifacts_v2_tags")]
        fk_names = [
            fk["name"] for fk in inspector.get_foreign_keys("artifacts_v2_tags")
        ]
        if "obj_id" in cols and "artifacts_v2_tags_obj_id_fkey" not in fk_names:
            op.create_foreign_key(
                "artifacts_v2_tags_obj_id_fkey",
                "artifacts_v2_tags",
                "artifacts_v2",
                ["obj_id"],
                ["id"],
            )

    # revert timestamp columns to DateTime
    tables = {
        "schedules_v2": ["next_run_time", "creation_time"],
        "runs_notifications": ["sent_time"],
        "runs": ["updated", "start_time"],
        "projects": ["created"],
        "pagination_cache": ["last_accessed"],
        "model_endpoints": ["updated", "created"],
        "hub_sources": ["updated", "created"],
        "functions": ["updated"],
        "feature_vectors": ["updated", "created"],
        "feature_sets": ["updated", "created"],
        "data_versions": ["created"],
        "background_tasks": ["updated", "created"],
        "artifacts_v2": ["updated", "created"],
        "artifacts": ["updated"],
        "alert_states": ["last_updated", "created"],
        "alert_configs_notifications": ["sent_time"],
    }
    for tbl, cols in tables.items():
        if tbl in inspector.get_table_names():
            for col in cols:
                op.alter_column(
                    tbl,
                    col,
                    existing_type=mlrun.db.sql_types.MicroSecondDateTime(),
                    type_=mlrun.db.sql_types.DateTime(),
                    existing_nullable=True,
                )
