# Copyright 2026 Iguazio
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

"""Index runs project iteration state start_time

Revision ID: 7bbf7ca69cb8
Revises: 9cae5c29c395
Create Date: 2026-06-10 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "7bbf7ca69cb8"
down_revision = "9cae5c29c395"
branch_labels = None
depends_on = None

_index_name = "idx_runs_project_iter_state_start"
_table_name = "runs"


def upgrade():
    inspector = sa.inspect(op.get_bind())
    existing = {idx["name"] for idx in inspector.get_indexes(_table_name)}
    if _index_name in existing:
        return
    op.create_index(
        _index_name,
        _table_name,
        ["project", "iteration", "state", "start_time"],
        unique=False,
    )


def downgrade():
    op.drop_index(_index_name, table_name=_table_name)
