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
"""notifications params to secret_params

Revision ID: eefc169f7633
Revises: 026c947c4487
Create Date: 2023-08-29 10:30:57.901466

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "eefc169f7633"
down_revision = "026c947c4487"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "runs_notifications",
        "params",
        nullable=True,
        new_column_name="secret_params",
        type_=sa.JSON(),
    )
    (
        op.add_column(
            "runs_notifications", sa.Column("params", sa.JSON(), nullable=True)
        ),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "runs_notifications",
        "params_secret",
        nullable=True,
        new_column_name="params",
        type_=sa.JSON(),
    )
    op.drop_column("runs_notifications", "params_secret")
    # ### end Alembic commands ###
