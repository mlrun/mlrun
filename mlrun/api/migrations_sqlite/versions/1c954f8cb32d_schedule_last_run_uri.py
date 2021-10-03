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
