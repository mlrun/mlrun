# Copyright 2023 Iguazio
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
"""init

Revision ID: c4af40b0bf61
Revises:
Create Date: 2021-09-30 10:55:51.956636

"""

import sqlalchemy as sa
import sqlalchemy.dialects.mysql
from alembic import op

from server.api.utils.db.sql_collation import SQLCollationUtil

# revision identifiers, used by Alembic.
revision = "c4af40b0bf61"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "key",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("updated", sa.TIMESTAMP(), nullable=True),
        sa.Column("body", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_artifacts_pk"),
        sa.UniqueConstraint("uid", "project", "key", name="_artifacts_uc"),
    )
    op.create_table(
        "feature_sets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("created", sa.TIMESTAMP(), nullable=True),
        sa.Column("updated", sa.TIMESTAMP(), nullable=True),
        sa.Column(
            "state",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
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
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("created", sa.TIMESTAMP(), nullable=True),
        sa.Column("updated", sa.TIMESTAMP(), nullable=True),
        sa.Column(
            "state",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
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
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("body", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.Column("updated", sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_functions_pk"),
        sa.UniqueConstraint("name", "project", "uid", name="_functions_uc"),
    )
    op.create_table(
        "logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("body", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_logs_pk"),
    )
    op.create_table(
        "marketplace_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("index", sa.Integer(), nullable=True),
        sa.Column("created", sa.TIMESTAMP(), nullable=True),
        sa.Column("updated", sa.TIMESTAMP(), nullable=True),
        sa.Column("object", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_marketplace_sources_pk"),
        sa.UniqueConstraint("name", name="_marketplace_sources_uc"),
    )
    op.create_table(
        "projects",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "description",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "owner",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "source",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("spec", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.Column("created", sa.TIMESTAMP(), nullable=True),
        sa.Column(
            "state",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id", name="_projects_pk"),
        sa.UniqueConstraint("name", name="_projects_uc"),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "uid",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("iteration", sa.Integer(), nullable=True),
        sa.Column(
            "state",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("body", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.Column("start_time", sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="_runs_pk"),
        sa.UniqueConstraint("uid", "project", "iteration", name="_runs_uc"),
    )
    op.create_table(
        "schedules_v2",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=False,
        ),
        sa.Column(
            "kind",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "desired_state",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "state",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("creation_time", sa.TIMESTAMP(), nullable=True),
        sa.Column(
            "cron_trigger_str",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "last_run_uri",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("struct", sqlalchemy.dialects.mysql.MEDIUMBLOB(), nullable=True),
        sa.Column("concurrency_limit", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="_schedules_v2_pk"),
        sa.UniqueConstraint("project", "name", name="_schedules_v2_uc"),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id", name="_users_pk"),
        sa.UniqueConstraint("name", name="_users_uc"),
    )
    op.create_table(
        "artifacts_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
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
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
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
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value_type",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["feature_set_id"], ["feature_sets.id"], name="_entities_feature_set_id_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_entities_pk"),
    )
    op.create_table(
        "feature_sets_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"], ["feature_sets.id"], name="_feature_sets_labels_parent_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_feature_sets_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_feature_sets_labels_uc"),
    )
    op.create_table(
        "feature_sets_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.Column(
            "obj_name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["obj_id"], ["feature_sets.id"], name="_feature_sets_tags_obj_id_fk"
        ),
        sa.ForeignKeyConstraint(
            ["obj_name"], ["feature_sets.name"], name="_feature_sets_tags_obj_name_fk"
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
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"], ["feature_vectors.id"], name="_feature_vectors_labels_parent_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_feature_vectors_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_feature_vectors_labels_uc"),
    )
    op.create_table(
        "feature_vectors_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.Column(
            "obj_name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["obj_id"], ["feature_vectors.id"], name="_feature_vectors_tags_obj_id_fk"
        ),
        sa.ForeignKeyConstraint(
            ["obj_name"],
            ["feature_vectors.name"],
            name="_feature_vectors_tags_obj_name_fk",
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
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value_type",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["feature_set_id"], ["feature_sets.id"], name="_features_feature_set_id_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_features_pk"),
    )
    op.create_table(
        "functions_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"], ["functions.id"], name="_functions_labels_parent_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_functions_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_functions_labels_uc"),
    )
    op.create_table(
        "functions_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.Column(
            "obj_name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["obj_id"], ["functions.id"], name="_functions_tags_obj_id_fk"
        ),
        sa.ForeignKeyConstraint(
            ["obj_name"], ["functions.name"], name="_functions_tags_obj_name_fk"
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
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
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
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["parent"], ["runs.id"], name="_runs_labels_parent_fk"),
        sa.PrimaryKeyConstraint("id", name="_runs_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_runs_labels_uc"),
    )
    op.create_table(
        "runs_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "project",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["obj_id"], ["runs.id"], name="_runs_tags_obj_id_fk"),
        sa.PrimaryKeyConstraint("id", name="_runs_tags_pk"),
        sa.UniqueConstraint("project", "name", "obj_id", name="_runs_tags_uc"),
    )
    op.create_table(
        "schedules_v2_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"], ["schedules_v2.id"], name="_schedules_v2_labels_parent_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_schedules_v2_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_schedules_v2_labels_uc"),
    )
    op.create_table(
        "entities_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
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
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"], ["features.id"], name="_features_labels_parent_fk"
        ),
        sa.PrimaryKeyConstraint("id", name="_features_labels_pk"),
        sa.UniqueConstraint("name", "parent", name="_features_labels_uc"),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("features_labels")
    op.drop_table("entities_labels")
    op.drop_table("schedules_v2_labels")
    op.drop_table("runs_tags")
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
    op.drop_table("users")
    op.drop_table("schedules_v2")
    op.drop_table("runs")
    op.drop_table("projects")
    op.drop_table("marketplace_sources")
    op.drop_table("logs")
    op.drop_table("functions")
    op.drop_table("feature_vectors")
    op.drop_table("feature_sets")
    op.drop_table("artifacts")
    # ### end Alembic commands ###
