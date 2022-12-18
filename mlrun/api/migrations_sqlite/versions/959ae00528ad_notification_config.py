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

"""notification config

Revision ID: 959ae00528ad
Revises: 6401142f2d7c
Create Date: 2022-09-20 10:40:41.354209

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "959ae00528ad"
down_revision = "6401142f2d7c"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "notifications",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project", sa.String(length=255)),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=255), nullable=False),
        sa.Column("message", sa.String(length=255), nullable=False),
        sa.Column("severity", sa.String(length=255), nullable=False),
        sa.Column("when", sa.String(length=255), nullable=False),
        sa.Column("condition", sa.String(length=255), nullable=False),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("run", sa.Integer(), nullable=True),
        sa.Column("sent_time", sa.TIMESTAMP(), nullable=True),
        sa.Column("status", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["run"],
            ["runs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "run", name="_notifications_uc"),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("notifications")
    # ### end Alembic commands ###
