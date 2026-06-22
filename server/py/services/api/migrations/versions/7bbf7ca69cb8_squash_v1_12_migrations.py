# Copyright 2024 Iguazio
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

"""squash v1.12 migrations

Revision ID: 7bbf7ca69cb8
Revises: 4a4172268db0
Create Date: 2026-06-17 11:43:54.018360

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "7bbf7ca69cb8"
down_revision = "4a4172268db0"
branch_labels = None
depends_on = None

_functions_table = "functions"
_functions_index = "idx_function_project_kind"

_runs_table = "runs"
# Covers the no-status-filter view; strict superset of the superseded index below.
_covering_no_state = "idx_runs_project_iter_start_updated"
# Covers the status-filtered view (state as equality column before the start_time range).
_covering_with_state = "idx_runs_project_iter_state_start_updated"
# (project, iteration, start_time, name) - now a prefix of _covering_no_state.
_superseded = "idx_runs_project_iter_start"


def upgrade():
    inspector = sa.inspect(op.get_bind())

    functions_indexes = {idx["name"] for idx in inspector.get_indexes(_functions_table)}
    if _functions_index not in functions_indexes:
        op.create_index(
            _functions_index, _functions_table, ["project", "kind"], unique=False
        )

    runs_indexes = {idx["name"] for idx in inspector.get_indexes(_runs_table)}
    if _covering_no_state not in runs_indexes:
        op.create_index(
            _covering_no_state,
            _runs_table,
            ["project", "iteration", "start_time", "name", "updated"],
            unique=False,
        )
    if _covering_with_state not in runs_indexes:
        op.create_index(
            _covering_with_state,
            _runs_table,
            ["project", "iteration", "state", "start_time", "name", "updated"],
            unique=False,
        )
    if _superseded in runs_indexes:
        op.drop_index(_superseded, table_name=_runs_table)


def downgrade():
    inspector = sa.inspect(op.get_bind())

    runs_indexes = {idx["name"] for idx in inspector.get_indexes(_runs_table)}
    if _superseded not in runs_indexes:
        op.create_index(
            _superseded,
            _runs_table,
            ["project", "iteration", "start_time", "name"],
            unique=False,
        )
    for name in (_covering_with_state, _covering_no_state):
        if name in runs_indexes:
            op.drop_index(name, table_name=_runs_table)

    functions_indexes = {idx["name"] for idx in inspector.get_indexes(_functions_table)}
    if _functions_index in functions_indexes:
        op.drop_index(_functions_index, table_name=_functions_table)
