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
Revises: eefc169f7633
Create Date: 2023-09-10 12:55:27.620429

"""

import sqlalchemy as sa
from alembic import op

from server.api.utils.db.sql_collation import SQLCollationUtil

# revision identifiers, used by Alembic.
revision = "27ed4ecb734c"
down_revision = "eefc169f7633"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "runs_notifications",
        sa.Column(
            "reason",
            sa.String(length=255, collation=SQLCollationUtil.collation()),
            nullable=True,
        ),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("runs_notifications", "reason")
    # ### end Alembic commands ###
