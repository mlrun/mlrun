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
"""add pagination cache

Revision ID: 3e81ad4e5ebf
Revises: c0e342d73bd0
Create Date: 2024-03-26 09:58:40.044072

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = "3e81ad4e5ebf"
down_revision = "c0e342d73bd0"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "pagination_cache",
        sa.Column("key", sa.String(length=255, collation="utf8_bin"), nullable=False),
        sa.Column("user", sa.String(length=255, collation="utf8_bin"), nullable=True),
        sa.Column(
            "function", sa.String(length=255, collation="utf8_bin"), nullable=True
        ),
        sa.Column("current_page", sa.Integer(), nullable=True),
        sa.Column("kwargs", sa.JSON(), nullable=True),
        sa.Column("last_accessed", mysql.TIMESTAMP(fsp=3), nullable=True),
        sa.PrimaryKeyConstraint("key"),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("pagination_cache")
    # ### end Alembic commands ###
