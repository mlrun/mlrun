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
"""Increase timestamp fields precision

Revision ID: 32bae1b0e29c
Revises: b86f5b53f3d7
Create Date: 2022-01-16 19:32:08.676120

"""

import sqlalchemy.dialects.mysql
from alembic import op

# revision identifiers, used by Alembic.
revision = "32bae1b0e29c"
down_revision = "b86f5b53f3d7"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        table_name="artifacts",
        column_name="updated",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="functions",
        column_name="updated",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="runs",
        column_name="start_time",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="runs",
        column_name="updated",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="schedules_v2",
        column_name="creation_time",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="projects",
        column_name="created",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="feature_sets",
        column_name="created",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="feature_sets",
        column_name="updated",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="feature_vectors",
        column_name="created",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="feature_vectors",
        column_name="updated",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="marketplace_sources",
        column_name="created",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="marketplace_sources",
        column_name="updated",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    op.alter_column(
        table_name="data_versions",
        column_name="created",
        type_=sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        table_name="artifacts", column_name="updated", type_=sqlalchemy.TIMESTAMP
    )
    op.alter_column(
        table_name="functions", column_name="updated", type_=sqlalchemy.TIMESTAMP
    )
    op.alter_column(
        table_name="runs", column_name="start_time", type_=sqlalchemy.TIMESTAMP
    )
    op.alter_column(
        table_name="runs", column_name="updated", type_=sqlalchemy.TIMESTAMP
    )
    op.alter_column(
        table_name="schedules_v2",
        column_name="creation_time",
        type_=sqlalchemy.TIMESTAMP,
    )
    op.alter_column(
        table_name="projects", column_name="created", type_=sqlalchemy.TIMESTAMP
    )
    op.alter_column(
        table_name="feature_sets", column_name="created", type_=sqlalchemy.TIMESTAMP
    )
    op.alter_column(
        table_name="feature_sets", column_name="updated", type_=sqlalchemy.TIMESTAMP
    )
    op.alter_column(
        table_name="feature_vectors", column_name="created", type_=sqlalchemy.TIMESTAMP
    )
    op.alter_column(
        table_name="feature_vectors", column_name="updated", type_=sqlalchemy.TIMESTAMP
    )
    op.alter_column(
        table_name="marketplace_sources",
        column_name="created",
        type_=sqlalchemy.TIMESTAMP,
    )
    op.alter_column(
        table_name="marketplace_sources",
        column_name="updated",
        type_=sqlalchemy.TIMESTAMP,
    )
    op.alter_column(
        table_name="data_versions", column_name="created", type_=sqlalchemy.TIMESTAMP
    )
    # ### end Alembic commands ###
