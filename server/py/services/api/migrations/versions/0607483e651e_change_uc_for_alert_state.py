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

"""change uc for alert_state

Revision ID: 0607483e651e
Revises: 57d26493fbff
Create Date: 2025-01-02 15:08:40.096362
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0607483e651e"
down_revision = "57d26493fbff"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    dialect = bind.dialect.name

    # On MySQL, drop the foreign key first so we can drop the index
    if dialect == "mysql":
        op.drop_constraint("alert_states_ibfk_1", "alert_states", type_="foreignkey")

    # Drop the old unique: as a constraint on Postgres, as an index elsewhere
    if dialect == "postgresql":
        op.drop_constraint("alert_states_uc", "alert_states", type_="unique")
    else:
        op.drop_index("alert_states_uc", table_name="alert_states")

    # Create the new unique constraint on parent_id
    op.create_unique_constraint("_alert_state_parent_uc", "alert_states", ["parent_id"])

    # Recreate the foreign key (MySQL needs this; Postgres will ignore duplicate names)
    op.create_foreign_key(
        "alert_states_ibfk_1",
        "alert_states",
        "alert_configs",
        ["parent_id"],
        ["id"],
    )


def downgrade():
    # Drop the FK temporarily
    op.drop_constraint("alert_states_ibfk_1", "alert_states", type_="foreignkey")

    # Drop our new unique constraint
    op.drop_constraint("_alert_state_parent_uc", "alert_states", type_="unique")

    # Re-create the old single-column index
    op.create_index(
        "alert_states_uc", "alert_states", ["id", "parent_id"], unique=False
    )

    # Recreate the FK
    op.create_foreign_key(
        "alert_states_ibfk_1",
        "alert_states",
        "alert_configs",
        ["parent_id"],
        ["id"],
    )
