# Copyright 2018 Iguazio
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
"""Adding feature labels

Revision ID: f7b5a1a03629
Revises: 2b6d23c715aa
Create Date: 2020-11-09 11:19:51.472174

"""
import sqlalchemy as sa
from alembic import op

from mlrun.api.utils.db.sql_collation import SQLCollationUtil

# revision identifiers, used by Alembic.
revision = "f7b5a1a03629"
down_revision = "2b6d23c715aa"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "entities_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"],
            ["entities.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "parent", name="_entities_labels_uc"),
    )
    op.create_table(
        "features_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "name",
            sa.String(255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column(
            "value",
            sa.String(255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"],
            ["features.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "parent", name="_features_labels_uc"),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("features_labels")
    op.drop_table("entities_labels")
    # ### end Alembic commands ###
