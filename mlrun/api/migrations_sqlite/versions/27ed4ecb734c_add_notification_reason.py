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
"""add notifications reason

Revision ID: 27ed4ecb734c
Revises: 114b2c80710f
Create Date: 2023-09-10 12:55:27.620429

"""

import sqlalchemy as sa
from alembic import op

import mlrun.api.utils.db.sql_collation

# revision identifiers, used by Alembic.
revision = "27ed4ecb734c"
down_revision = "114b2c80710f"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("runs_notifications") as batch_op:
        batch_op.add_column(
            sa.Column(
                "reason",
                sa.String(
                    length=255,
                    collation=mlrun.api.utils.db.sql_collation.SQLCollationUtil.collation(),
                ),
                nullable=True,
            )
        )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("runs_notifications") as batch_op:
        batch_op.drop_column("reason")
    # ### end Alembic commands ###
