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
"""Schedule last run uri

Revision ID: 1c954f8cb32d
Revises: f7b5a1a03629
Create Date: 2020-11-11 09:39:09.551025

"""
import sqlalchemy as sa
from alembic import op

from mlrun.api.utils.db.sql_collation import SQLCollationUtil

# revision identifiers, used by Alembic.
revision = "1c954f8cb32d"
down_revision = "f7b5a1a03629"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("schedules_v2") as batch_op:
        batch_op.add_column(
            sa.Column(
                "last_run_uri",
                sa.String(255, collation=SQLCollationUtil.collation()),
                nullable=True,
            )
        )


def downgrade():
    with op.batch_alter_table("schedules_v2") as batch_op:
        batch_op.drop_column("last_run_uri")
