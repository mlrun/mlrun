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
"""Refactoring feature set

Revision ID: 863114f0c659
Revises: f7b5a1a03629
Create Date: 2020-11-11 11:22:36.653049

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "863114f0c659"
down_revision = "1c954f8cb32d"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("feature_sets") as batch_op:
        batch_op.add_column(sa.Column("object", sa.JSON(), nullable=True))
        batch_op.drop_column("status")


def downgrade():
    with op.batch_alter_table("feature_sets") as batch_op:
        batch_op.add_column(sa.Column("status", sa.JSON(), nullable=True))
        batch_op.drop_column("object")
