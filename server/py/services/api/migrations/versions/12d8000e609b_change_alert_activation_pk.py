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

"""Change alert activation PK

Revision ID: 12d8000e609b
Revises: 650f0ce2da6f
Create Date: 2024-11-13 13:45:08.512249
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "12d8000e609b"
down_revision = "650f0ce2da6f"
branch_labels = None
depends_on = None


def upgrade():
    # drop old PK and recreate with (id, activation_time)
    op.drop_constraint("_alert_activation_uc", "alert_activations", type_="primary")
    op.create_primary_key(
        "_alert_activation_uc", "alert_activations", ["id", "activation_time"]
    )

    conn = op.get_bind()
    dialect = conn.dialect.name

    if dialect == "mysql":
        # MySQL auto_increment
        op.execute("""
            ALTER TABLE alert_activations
            MODIFY COLUMN id INT NOT NULL AUTO_INCREMENT
        """)
    elif dialect == "postgresql":
        # PostgreSQL: create a sequence and set default
        op.execute("""
            CREATE SEQUENCE IF NOT EXISTS alert_activations_id_seq
              OWNED BY alert_activations.id;
            ALTER TABLE alert_activations
              ALTER COLUMN id SET DEFAULT nextval('alert_activations_id_seq');
        """)


def downgrade():
    conn = op.get_bind()
    dialect = conn.dialect.name

    # revert the id default/auto-increment
    if dialect == "mysql":
        op.execute("""
            ALTER TABLE alert_activations
            MODIFY COLUMN id INT NOT NULL
        """)
    elif dialect == "postgresql":
        op.execute("""
            ALTER TABLE alert_activations
              ALTER COLUMN id DROP DEFAULT;
            DROP SEQUENCE IF EXISTS alert_activations_id_seq;
        """)

    # drop current PK and restore original (activation_time, id)
    op.drop_constraint("_alert_activation_uc", "alert_activations", type_="primary")
    op.create_primary_key(
        "_alert_activation_uc", "alert_activations", ["activation_time", "id"]
    )
