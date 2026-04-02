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
#
"""Squashed init — full schema as of 1.7

This migration squashes all 27 migrations from the original init (c4af40b0bf61)
through the 1.7 HEAD (fcf2ea01f99a_index_artifacts_v2_project_and_kind).

The original init migration created FK constraints on non-unique columns
(feature_sets.name, feature_vectors.name, functions.name), which broke
MySQL 8.4+ with ER_FK_NO_INDEX_PARENT (error 6125). Those FKs were dropped
two migrations later by 4903aef6a91d and were never valid. This squashed
migration simply omits them from the start.

Existing databases at revision fcf2ea01f99a are unaffected — Alembic sees
their alembic_version and skips this migration entirely.

Revision ID: fcf2ea01f99a
Revises:
Create Date: 2024-07-30 22:07:05.051576

"""

import sqlalchemy as sa
import sqlalchemy.dialects.mysql
from alembic import op
from sqlalchemy.dialects import mysql

from server.api.utils.db.sql_types import SQLTypesUtil

# revision identifiers, used by Alembic.
revision = "fcf2ea01f99a"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### Squashed schema — all tables at 1.7, invalid FKs omitted ###

    # ── Independent base tables ──────────────────────────────────────────────

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "key",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("updated", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column("body", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_artifacts_pk"),
        sa.UniqueConstraint("uid", "project", "key", name="_artifacts_uc"),
    )
    op.create_table(
        "feature_sets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("created", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column("updated", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column(
            "state",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("object", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_feature_sets_pk"),
        sa.UniqueConstraint("name", "project", "uid", name="_feature_set_uc"),
    )
    op.create_table(
        "feature_vectors",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("created", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column("updated", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column(
            "state",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("object", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_feature_vectors_pk"),
        sa.UniqueConstraint("name", "project", "uid", name="_feature_vectors_uc"),
    )
    op.create_table(
        "functions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("body", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.Column("updated", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_functions_pk"),
        sa.UniqueConstraint("name", "project", "uid", name="_functions_uc"),
    )
    op.create_table(
        "logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("body", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_logs_pk"),
    )
    # Renamed from marketplace_sources in 28383af526f3; UC renamed in cfe2a22173fc.
    op.create_table(
        "hub_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("index", sa.Integer(), nullable=True),
        sa.Column("created", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column("updated", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column("object", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_marketplace_sources_pk"),
        sa.UniqueConstraint("name", name="_hub_sources_uc"),
    )
    op.create_table(
        "projects",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "description",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "owner",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "source",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("spec", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.Column("created", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column(
            "state",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("default_function_node_selector", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_projects_pk"),
        sa.UniqueConstraint("name", name="_projects_uc"),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("iteration", sa.Integer(), nullable=True),
        sa.Column(
            "state",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("body", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.Column(
            "start_time", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "updated", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True
        ),
        sa.Column("requested_logs", sa.BOOLEAN(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_runs_pk"),
        sa.UniqueConstraint("uid", "project", "iteration", name="_runs_uc"),
    )
    op.create_table(
        "schedules_v2",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=False,
        ),
        sa.Column(
            "kind",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "desired_state",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "state",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "creation_time",
            sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
            nullable=True,
        ),
        sa.Column(
            "cron_trigger_str",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "last_run_uri",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("struct", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.Column("concurrency_limit", sa.Integer(), nullable=False),
        sa.Column(
            "next_run_time",
            sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id", name="_schedules_v2_pk"),
        sa.UniqueConstraint("project", "name", name="_schedules_v2_uc"),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id", name="_users_pk"),
        sa.UniqueConstraint("name", name="_users_uc"),
    )
    op.create_table(
        "data_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(255, collation="utf8_bin"), nullable=True),
        sa.Column(
            "created", sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3), nullable=True
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version", name="_versions_uc"),
    )
    op.create_table(
        "background_tasks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "project", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column("created", mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column("updated", mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column(
            "state", sa.String(length=255, collation="utf8_bin"), nullable=True
        ),
        sa.Column("timeout", sa.Integer(), nullable=True),
        sa.Column(
            "error", sa.String(length=255, collation="utf8_bin"), nullable=True
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "project", name="_background_tasks_uc"),
    )
    op.create_table(
        "datastore_profiles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name", sa.String(length=255, collation="utf8_bin"), nullable=True
        ),
        sa.Column(
            "project", sa.String(length=255, collation="utf8_bin"), nullable=True
        ),
        sa.Column(
            "type", sa.String(length=255, collation="utf8_bin"), nullable=True
        ),
        sa.Column("object", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "project", name="_datastore_profiles_uc"),
    )
    op.create_table(
        "artifacts_v2",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "key",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "kind",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "producer_id",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("iteration", sa.Integer(), nullable=True),
        sa.Column("best_iteration", sa.BOOLEAN(), nullable=True, index=True),
        sa.Column("object", mysql.MEDIUMBLOB(), nullable=True),
        sa.Column("created", mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column("updated", mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("uid", "project", "key", name="_artifacts_v2_uc"),
    )
    op.create_table(
        "pagination_cache",
        sa.Column(
            "key", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "user", sa.String(length=255, collation="utf8_bin"), nullable=True
        ),
        sa.Column(
            "function", sa.String(length=255, collation="utf8_bin"), nullable=True
        ),
        sa.Column("current_page", sa.Integer(), nullable=True),
        sa.Column("page_size", sa.Integer(), nullable=True),
        sa.Column("kwargs", sa.JSON(), nullable=True),
        sa.Column("last_accessed", mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_table(
        "alert_configs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "project", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column("object", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project", "name", name="_alert_configs_uc"),
    )
    op.create_table(
        "alert_templates",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column("object", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="_alert_templates_uc"),
    )
    op.create_table(
        "project_summaries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project", sa.String(length=255, collation="utf8mb3_bin"), nullable=False
        ),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("updated", mysql.DATETIME(timezone=True, fsp=3), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project", name="_project_summaries_uc"),
    )
    op.create_table(
        "time_window_trackers",
        sa.Column(
            "key", sa.String(length=255, collation="utf8mb3_bin"), nullable=False
        ),
        sa.Column(
            "timestamp", mysql.DATETIME(timezone=True, fsp=3), nullable=False
        ),
        sa.Column("max_window_size_seconds", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("key"),
    )

    # ── Tables with FK dependencies ───────────────────────────────────────────

    op.create_table(
        "artifacts_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"], ["artifacts.id"], name="_artifacts_labels_paren_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_artifacts_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_artifacts_labels_uc"),
    )
    op.create_table(
        "artifacts_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["obj_id"], ["artifacts.id"], name="_artifacts_tags_obj_id_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_artifacts_tags_pk"),
        sa.UniqueConstraint("project", "name", "obj_id", name="_artifacts_tags_uc"),
    )
    op.create_table(
        "entities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("feature_set_id", sa.Integer(), nullable=True),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value_type",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["feature_set_id"],
            ["feature_sets.id"],
            name="_entities_feature_set_id_fk",
        ),
        sa.PrimaryKeyConstraint("id", name="_entities_pk"),
    )
    op.create_table(
        "feature_sets_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"],
            ["feature_sets.id"],
            name="_feature_sets_labels_parent_fk",
        ),
        sa.PrimaryKeyConstraint("id", name="_feature_sets_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_feature_sets_labels_uc"),
    )
    # No FK on obj_name — the original FK referenced a non-unique column and was
    # dropped by 4903aef6a91d two migrations after creation.
    op.create_table(
        "feature_sets_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.Column(
            "obj_name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["obj_id"], ["feature_sets.id"], name="_feature_sets_tags_obj_id_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_feature_sets_tags_pk"),
        sa.UniqueConstraint(
            "project", "name", "obj_name", name="_feature_sets_tags_uc"
        ),
    )
    op.create_table(
        "feature_vectors_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"],
            ["feature_vectors.id"],
            name="_feature_vectors_labels_parent_fk",
        ),
        sa.PrimaryKeyConstraint("id", name="_feature_vectors_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_feature_vectors_labels_uc"),
    )
    # No FK on obj_name — same reason as feature_sets_tags.
    op.create_table(
        "feature_vectors_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.Column(
            "obj_name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["obj_id"],
            ["feature_vectors.id"],
            name="_feature_vectors_tags_obj_id_fk",
        ),
        sa.PrimaryKeyConstraint("id", name="_feature_vectors_tags_pk"),
        sa.UniqueConstraint(
            "project", "name", "obj_name", name="_feature_vectors_tags_uc"
        ),
    )
    op.create_table(
        "features",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("feature_set_id", sa.Integer(), nullable=True),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value_type",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["feature_set_id"],
            ["feature_sets.id"],
            name="_features_feature_set_id_fk",
        ),
        sa.PrimaryKeyConstraint("id", name="_features_pk"),
    )
    op.create_table(
        "functions_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"], ["functions.id"], name="_functions_labels_parent_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_functions_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_functions_labels_uc"),
    )
    # No FK on obj_name — same reason as feature_sets_tags.
    op.create_table(
        "functions_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.Column(
            "obj_name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["obj_id"], ["functions.id"], name="_functions_tags_obj_id_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_functions_tags_pk"),
        sa.UniqueConstraint("project", "name", "obj_name", name="_functions_tags_uc"),
    )
    op.create_table(
        "project_users",
        sa.Column("project_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name="_project_users_project_id_fk"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="_project_users_user_id_fk"
        ),
    )
    op.create_table(
        "projects_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"], ["projects.id"], name="_projects_labels_parent_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_projects_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_projects_labels_uc"),
    )
    op.create_table(
        "runs_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"], ["runs.id"], name="_runs_labels_parent_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_runs_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_runs_labels_uc"),
    )
    op.create_table(
        "runs_notifications",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project", sa.String(length=255, collation="utf8_bin")),
        sa.Column(
            "name", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "kind", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "message", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "severity", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "when", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "condition", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column("secret_params", sa.JSON(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("sent_time", mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column(
            "status", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "reason",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["runs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "parent_id", name="_runs_notifications_uc"),
    )
    op.create_table(
        "runs_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["obj_id"], ["runs.id"], name="_runs_tags_obj_id_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_runs_tags_pk"),
        sa.UniqueConstraint("project", "name", "obj_id", name="_runs_tags_uc"),
    )
    op.create_table(
        "schedules_v2_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"],
            ["schedules_v2.id"],
            name="_schedules_v2_labels_parent_fk",
        ),
        sa.PrimaryKeyConstraint("id", name="_schedules_v2_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_schedules_v2_labels_uc"),
    )
    op.create_table(
        "entities_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"], ["entities.id"], name="_entities_labels_parent_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_entities_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_entities_labels_uc"),
    )
    op.create_table(
        "features_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"], ["features.id"], name="_features_labels_parent_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_features_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_features_labels_uc"),
    )
    op.create_table(
        "artifacts_v2_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"],
            ["artifacts_v2.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "parent", name="_artifacts_v2_labels_uc"),
    )
    op.create_table(
        "artifacts_v2_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "name",
            sa.String(255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.Column(
            "obj_name",
            sa.String(255, collation=SQLTypesUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["obj_id"],
            ["artifacts_v2.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project", "name", "obj_id", name="_artifacts_v2_tags_uc"),
    )
    op.create_table(
        "alert_configs_notifications",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project", sa.String(length=255, collation="utf8_bin"), nullable=True
        ),
        sa.Column(
            "name", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "kind", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "message", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "severity", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "when", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "condition", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column("secret_params", sa.JSON(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("sent_time", mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column(
            "status", sa.String(length=255, collation="utf8_bin"), nullable=False
        ),
        sa.Column(
            "reason", sa.String(length=255, collation="utf8_bin"), nullable=True
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["alert_configs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "name", "parent_id", name="_alert_configs_notifications_uc"
        ),
    )
    op.create_table(
        "alert_states",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=True),
        sa.Column("created", mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column("last_updated", mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.Column("active", sa.BOOLEAN(), nullable=True),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("object", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["alert_configs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "parent_id", name="alert_states_uc"),
    )

    # ── Indices ───────────────────────────────────────────────────────────────

    op.create_index("ix_runs_requested_logs", "runs", ["requested_logs"], unique=False)
    op.create_index("idx_runs_project_id", "runs", ["id", "project"], unique=True)
    op.create_index(
        "idx_artifacts_labels_name_value",
        "artifacts_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_entities_labels_name_value",
        "entities_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_feature_sets_labels_name_value",
        "feature_sets_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_projects_labels_name_value",
        "projects_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_functions_labels_name_value",
        "functions_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_features_labels_name_value",
        "features_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_feature_vectors_labels_name_value",
        "feature_vectors_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_runs_labels_name_value", "runs_labels", ["name", "value"], unique=False
    )
    op.create_index(
        "idx_schedules_v2_labels_name_value",
        "schedules_v2_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(op.f("ix_artifacts_v2_key"), "artifacts_v2", ["key"], unique=False)
    op.create_index(
        "idx_artifacts_producer_id_best_iteration_and_project",
        "artifacts_v2",
        ["project", "producer_id", "best_iteration"],
        unique=False,
    )
    op.create_index(
        "idx_artifacts_v2_tags_project_name_obj_name",
        "artifacts_v2_tags",
        ["project", "name", "obj_name"],
        unique=False,
    )
    op.create_index(
        "idx_artifacts_v2_labels_name_value",
        "artifacts_v2_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_project_kind", "artifacts_v2", ["project", "kind"], unique=False
    )
    # ### end Alembic commands ###


def downgrade():
    # ### Drop in reverse dependency order ###
    op.drop_index("idx_project_kind", table_name="artifacts_v2")
    op.drop_index("idx_artifacts_v2_labels_name_value", table_name="artifacts_v2_labels")
    op.drop_index(
        "idx_artifacts_v2_tags_project_name_obj_name", table_name="artifacts_v2_tags"
    )
    op.drop_index(
        "idx_artifacts_producer_id_best_iteration_and_project",
        table_name="artifacts_v2",
    )
    op.drop_index(op.f("ix_artifacts_v2_key"), table_name="artifacts_v2")
    op.drop_index("idx_schedules_v2_labels_name_value", table_name="schedules_v2_labels")
    op.drop_index("idx_runs_labels_name_value", table_name="runs_labels")
    op.drop_index(
        "idx_feature_vectors_labels_name_value", table_name="feature_vectors_labels"
    )
    op.drop_index("idx_features_labels_name_value", table_name="features_labels")
    op.drop_index("idx_functions_labels_name_value", table_name="functions_labels")
    op.drop_index("idx_projects_labels_name_value", table_name="projects_labels")
    op.drop_index(
        "idx_feature_sets_labels_name_value", table_name="feature_sets_labels"
    )
    op.drop_index("idx_entities_labels_name_value", table_name="entities_labels")
    op.drop_index("idx_artifacts_labels_name_value", table_name="artifacts_labels")
    op.drop_index("idx_runs_project_id", table_name="runs")
    op.drop_index("ix_runs_requested_logs", table_name="runs")

    op.drop_table("alert_states")
    op.drop_table("alert_configs_notifications")
    op.drop_table("artifacts_v2_tags")
    op.drop_table("artifacts_v2_labels")
    op.drop_table("features_labels")
    op.drop_table("entities_labels")
    op.drop_table("schedules_v2_labels")
    op.drop_table("runs_tags")
    op.drop_table("runs_notifications")
    op.drop_table("runs_labels")
    op.drop_table("projects_labels")
    op.drop_table("project_users")
    op.drop_table("functions_tags")
    op.drop_table("functions_labels")
    op.drop_table("features")
    op.drop_table("feature_vectors_tags")
    op.drop_table("feature_vectors_labels")
    op.drop_table("feature_sets_tags")
    op.drop_table("feature_sets_labels")
    op.drop_table("entities")
    op.drop_table("artifacts_tags")
    op.drop_table("artifacts_labels")
    op.drop_table("alert_configs")
    op.drop_table("alert_templates")
    op.drop_table("artifacts_v2")
    op.drop_table("pagination_cache")
    op.drop_table("project_summaries")
    op.drop_table("time_window_trackers")
    op.drop_table("datastore_profiles")
    op.drop_table("background_tasks")
    op.drop_table("data_versions")
    op.drop_table("users")
    op.drop_table("schedules_v2")
    op.drop_table("runs")
    op.drop_table("projects")
    op.drop_table("hub_sources")
    op.drop_table("logs")
    op.drop_table("functions")
    op.drop_table("feature_vectors")
    op.drop_table("feature_sets")
    op.drop_table("artifacts")
    # ### end Alembic commands ###
