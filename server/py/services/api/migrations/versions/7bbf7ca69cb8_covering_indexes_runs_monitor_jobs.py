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

"""Covering indexes for runs Monitor Jobs query

Adds covering indexes so the "latest run per (project, name)" query
(row_number() OVER (PARTITION BY project, name ORDER BY updated)) can run
index-only instead of a clustered-index lookup per row - ML-12590.

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

_table_name = "runs"
# Covers the no-status-filter view; strict superset of the superseded index below.
_covering_no_state = "idx_runs_project_iter_start_updated"
# Covers the status-filtered view (state as equality column before the start_time range).
_covering_with_state = "idx_runs_project_iter_state_start_updated"
# (project, iteration, start_time, name) - now a prefix of _covering_no_state.
_superseded = "idx_runs_project_iter_start"


def upgrade():
    inspector = sa.inspect(op.get_bind())
    existing = {idx["name"] for idx in inspector.get_indexes(_table_name)}
    if _covering_no_state not in existing:
        op.create_index(
            _covering_no_state,
            _table_name,
            ["project", "iteration", "start_time", "name", "updated"],
            unique=False,
        )
    if _covering_with_state not in existing:
        op.create_index(
            _covering_with_state,
            _table_name,
            ["project", "iteration", "state", "start_time", "name", "updated"],
            unique=False,
        )
    if _superseded in existing:
        op.drop_index(_superseded, table_name=_table_name)


def downgrade():
    inspector = sa.inspect(op.get_bind())
    existing = {idx["name"] for idx in inspector.get_indexes(_table_name)}
    if _superseded not in existing:
        op.create_index(
            _superseded,
            _table_name,
            ["project", "iteration", "start_time", "name"],
            unique=False,
        )
    for name in (_covering_with_state, _covering_no_state):
        if name in existing:
            op.drop_index(name, table_name=_table_name)
